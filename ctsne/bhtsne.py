#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
A simple Python wrapper for the bh_tsne binary that makes it easier to use it
for TSV files in a pipeline without any shell script trickery.

Note: The script does some minimal sanity checking of the input, but don't
    expect it to cover all cases. After all, it is a just a wrapper.

Example:

    > echo -e '1.0\t0.0\n0.0\t1.0' | ./bhtsne.py -d 2 -p 0.1
    -2458.83181442  -6525.87718385
    2458.83181442   6525.87718385

The output will not be normalised, maybe the below one-liner is of interest?:

    python -c 'import numpy;  from sys import stdin, stdout;
        d = numpy.loadtxt(stdin); d -= d.min(axis=0); d /= d.max(axis=0);
        numpy.savetxt(stdout, d, fmt="%.8f", delimiter="\t")'

Authors:     Pontus Stenetorp    <pontus stenetorp se>
             Philippe Remy       <github: philipperemy>
Version:    2016-03-08
'''

# Copyright (c) 2013, Pontus Stenetorp <pontus stenetorp se>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
import io
import os
import sys
import subprocess

from os import devnull
from os.path import abspath, dirname, isfile, join as path_join
from sys import stderr, stdin, stdout
from argparse import ArgumentParser, FileType
from platform import system
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from tempfile import mkdtemp

import numpy as np



### Constants
IS_WINDOWS = True if system() == 'Windows' else False
BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'windows', 'bh_tsne.exe') if IS_WINDOWS else path_join(dirname(__file__), 'bh_tsne')
assert isfile(BH_TSNE_BIN_PATH), ('Unable to find the bh_tsne binary in the '
    'same directory as this script, have you forgotten to compile it?: {}'
    ).format(BH_TSNE_BIN_PATH)
# Default hyper-parameter values from van der Maaten (2014)
# https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf (Experimental Setup, page 13)
DEFAULT_NO_DIMS = 2
INITIAL_DIMENSIONS = 50
DEFAULT_PERPLEXITY = 50
DEFAULT_THETA = 0.5
EMPTY_SEED = -1
DEFAULT_USE_PCA = False
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_K = 3

###

def _argparse():
    argparse = ArgumentParser('Conditional bh_tsne Python wrapper')
    argparse.add_argument('-d', '--no_dims', type=int,
                          default=DEFAULT_NO_DIMS)
    argparse.add_argument('-p', '--perplexity', type=float,
            default=DEFAULT_PERPLEXITY)
    # 0.0 for theta is equivalent to vanilla t-SNE
    argparse.add_argument('-t', '--theta', type=float, default=DEFAULT_THETA)
    argparse.add_argument('-r', '--seed', type=int, default=EMPTY_SEED)
    argparse.add_argument('-n', '--initial_dims', type=int, default=INITIAL_DIMENSIONS)
    argparse.add_argument('-v', '--verbose', action='store_true')
    argparse.add_argument('-i', '--input', type=FileType('r'), default=stdin)
    argparse.add_argument('-o', '--output', type=FileType('w'),
            default=stdout)
    argparse.add_argument('--use_pca', action='store_true')
    argparse.add_argument('--no_pca', dest='use_pca', action='store_false')
    argparse.set_defaults(use_pca=DEFAULT_USE_PCA)
    argparse.add_argument('-m', '--max_iter', type=int, default=DEFAULT_MAX_ITERATIONS)
    argparse.add_argument('-k', '--keep_nonzero', type=int, default=DEFAULT_K)

    return argparse


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


def _is_filelike_object(f):
    try:
        return isinstance(f, (file, io.IOBase))
    except NameError:
        # 'file' is not a class in python3
        return isinstance(f, io.IOBase)


def init_bh_tsne(samples, labels, alpha, beta, num_label_vals, workdir, no_dims=DEFAULT_NO_DIMS, 
                 initial_dims=INITIAL_DIMENSIONS, perplexity=DEFAULT_PERPLEXITY,
                 theta=DEFAULT_THETA, seed=EMPTY_SEED, verbose=False, 
                 use_pca=DEFAULT_USE_PCA, max_iter=DEFAULT_MAX_ITERATIONS, keep_nonzero=DEFAULT_K):

    if use_pca:
        samples = samples - np.mean(samples, axis=0)
        cov_x = np.dot(np.transpose(samples), samples)
        [eig_val, eig_vec] = np.linalg.eig(cov_x)

        # sorting the eigen-values in the descending order
        eig_vec = eig_vec[:, eig_val.argsort()[::-1]]

        if initial_dims > len(eig_vec):
            initial_dims = len(eig_vec)

        # truncating the eigen-vectors matrix to keep the most important vectors
        eig_vec = np.real(eig_vec[:, :initial_dims])
        samples = np.dot(samples, eig_vec)

    # Assume that the dimensionality of the first sample is representative for
    #   the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)

    # Note: The binary format used by bh_tsne is roughly the same as for
    #   vanilla tsne
    with open(path_join(workdir, 'data.dat'), 'wb') as data_file:
        # Write the bh_tsne header
        data_file.write(pack('ddddiiiiii', alpha, beta, theta, perplexity, num_label_vals, 
                             sample_count, sample_dim, no_dims, max_iter, keep_nonzero))

        # Write labels
        data_file.write(pack('{}i'.format(len(labels)), *labels))
        # Then write the data
        for sample in samples:
            data_file.write(pack('{}d'.format(len(sample)), *sample))
        # Write random seed if specified
        if seed != EMPTY_SEED:
            data_file.write(pack('i', seed))


def read_results(workdir):
    with open(path_join(workdir, 'result.dat'), 'rb') as output_file:
        # The first two integers are just the number of samples and the
        #   dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        # Collect the results, but they may be out of order
        results = [_read_unpack('{}d'.format(result_dims), output_file)
            for _ in range(result_samples)]
        # Now collect the landmark data so that we can return the data in
        #   the order it arrived
        results = [(_read_unpack('i', output_file), e) for e in results]
        # Put the results in order and yield it
        results.sort()
        for _, result in results:
            yield result
        # The last piece of data is the cost for each sample, we ignore it
        #read_unpack('{}d'.format(sample_count), output_file)


def bh_tsne(workdir, verbose=False):
    try:
        result = subprocess.check_output(
            [abspath(BH_TSNE_BIN_PATH)],
            cwd=workdir
        )
        if verbose:
            result = result.decode()
            print(result)
    except subprocess.CalledProcessError as grepexc:
        print(
            f"ct-SNE did not execute correctly. Error {grepexc.returncode}, Result {grepexc.output.decode()}"
        )
        


def run_bh_tsne(data, labels, alpha=None, beta=None, no_dims=2, perplexity=50,
                theta=0.5, seed=-1, verbose=False, initial_dims=50,
                use_pca=True, max_iter=1000, keep_nonzero=3):
    '''
    Run TSNE based on the Barnes-HT algorithm

    Parameters:
    ----------
    data: file or numpy.array
        The data used to run TSNE, one sample per row
    no_dims: int
    perplexity: int
    seed: int
    theta: float
    initial_dims: int
    verbose: boolean
    use_pca: boolean
    max_iter: int
    keep_nonzero: int (approximation of high-dimensional similarities keep keep_nonzero*perplexity neighbors)
    '''
    # check labels
    labels = np.array(labels).astype(np.int32)
    n = len(labels)
    unique_labels = np.unique(labels)
    num_label_vals = len(unique_labels)
    assert np.max(labels) < num_label_vals, f"Labels must be integers starting between 0 and {num_label_vals - 1}."
    assert np.min(labels) >= 0, "Labels must be integers starting from 0."

    # compute alpha and beta prime
    num_labels = [sum(labels == val) for val in unique_labels]
    ratio = sum(nl*(nl-1)/(n*(n-1)) for nl in num_labels)
    if not alpha and not beta:
        alpha = 1.0
        beta = 1.0
    elif not alpha:
        alpha = (1-beta*(1-ratio))/ratio
    elif not beta:
        beta = (1-alpha*ratio)/(1-ratio)
    else:
        if not np.isclose(alpha, (1-beta*(1-ratio))/ratio):
            print('alpha and beta violate P(\Delta | i, j) normalization constraint.')

    if verbose:
        print('alpha = {:.6f}, beta = {:.6f}'.format(alpha, beta))

    # bh_tsne works with fixed input and output paths, give it a temporary
    #   directory to work in so we don't clutter the filesystem
    tmp_dir_path = mkdtemp()

    if _is_filelike_object(data):
        data = np.loadtxt(data)
    else:
        data = np.asarray(data, dtype=np.float64)

    if keep_nonzero*perplexity > data.shape[0]:
        keep_nonzero = data.shape[0] // perplexity
        print(f"keep_nonzero was too large. Now using keep_nonzero = {keep_nonzero}")

    init_bh_tsne(data, labels, alpha, beta, num_label_vals,
                 tmp_dir_path, no_dims=no_dims, perplexity=perplexity,
                 theta=theta, seed=seed,verbose=verbose,
                 initial_dims=initial_dims, use_pca=use_pca,
                 max_iter=max_iter, keep_nonzero=keep_nonzero)
    bh_tsne(tmp_dir_path, verbose)
    res = []
    for result in read_results(tmp_dir_path):
        sample_res = []
        for r in result:
            sample_res.append(r)
        res.append(sample_res)
    rmtree(tmp_dir_path)
    return np.asarray(res, dtype='float64')
    

def main(args):
    parser = _argparse()

    if len(args) <= 1:
        print(parser.print_help())
        return

    argp = parser.parse_args(args[1:])

    for result in run_bh_tsne(argp.input, no_dims=argp.no_dims, perplexity=argp.perplexity, theta=argp.theta, seed=argp.seed,
            verbose=argp.verbose, initial_dims=argp.initial_dims, use_pca=argp.use_pca, max_iter=argp.max_iter, keep_nonzero=args.keep_nonzero):
        fmt = ''
        for i in range(1, len(result)):
            fmt = fmt + '{}\t'
        fmt = fmt + '{}\n'
        argp.output.write(fmt.format(*result))

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
