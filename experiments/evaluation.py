import numpy as np
import pandas as pd
import os
import subprocess
from tqdm import tqdm
import wandb
from sklearn.neighbors import NearestNeighbors


def compute_lisi(embedding_df, perplexity):
    """Evaluation metric: Local Inverse Simpsons Index using the scPOP
    package. https://rdrr.io/cran/scPOP/man/lisi.html.

    Args:
        embedding_df (data frame): data frame with embedding dimensions
            "x" and "y" and discrete feature columns for which to compute the LISI
        perplexity (int): Neighborhood size for the soft neighborhood calculation.

    Returns:
        data frame: each row contains the LISI values for a point from the
            embedding computed across all available labels in embedding_df.
    """
    print(f"Computing LISI with perplexity {perplexity}...")
    # store in tmp
    tmp_file = "tmp_embedding_df.csv"
    tmp_dest_file = "tmp_lisi.csv"

    embedding_df.to_csv(tmp_file, index=False)

    # call R script
    command = [
        "Rscript",
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "lisi.R"),
        tmp_file,
        tmp_dest_file,
        str(perplexity),
    ]

    _ = subprocess.check_output(command)
    lisi = pd.read_csv(tmp_dest_file)

    if os.path.isfile(tmp_dest_file):
        os.remove(tmp_dest_file)
    if os.path.isfile(tmp_file):
        os.remove(tmp_file)
    lisi = lisi.rename(columns=lambda c: "lisi_" + c)
    return lisi


def compute_laplacian(emb, labels, n_neigh, emb_nbrs, sample=None):
    """The laplacian score measures the fraction of points within a fixed-
    sized neighborhood that have a different label.

    Args:
        emb (ndarray): low dimensional embedding
        labels (ndarray): discrete labels to compute the laplacian
        n_neigh (int): locality of the score (number of neighbors to consider)
        emb_nbrs (sklearn.neighbors.NearestNeighbors): nearest neighbor search
            data structure fitted on the low-dimensional embedding
        sample (ndarray, optional): Indices of data subsample for faster
            computation. Defaults to None.

    Returns:
        ndarray: laplacian score for each point, -1 for points not in the sample
    """
    if sample is None:
        sample = np.arange(emb.shape[0])

    print(f"Computing laplacian scores with k={n_neigh} neighbors...", end="")
    # fraction of neighbors with different label
    lap = np.full(emb.shape[0], -1, dtype=float)
    for i in sample:
        emb_nn = emb_nbrs.kneighbors(
            emb[i, :].reshape(1, -1), n_neighbors=n_neigh + 1, return_distance=False
        )[0, 1:]
        diff_label = np.count_nonzero(labels[emb_nn] != labels[i])
        lap[i] = diff_label / n_neigh

    print("Done.")
    return lap


def get_random_laplacian(num_points, labels):
    """Average Laplacian score for random neighborhoods.

    Args:
        num_points (int): number of points in the dataset
        labels (ndarray): discrete labels for all points

    Returns:
        float: average laplacian
    """
    labels, label_counts = np.unique(labels, return_counts=True)
    random_lap = np.sum(
        [(c * (num_points - c)) / (num_points * (num_points - 1)) for c in label_counts]
    )
    return random_lap


def laplacian_curve(emb, labels, emb_nbrs, K_use, sample=None):
    """Coomputing the Laplacian on different neighborhood sizes.

    Args:
        emb (ndarray): low dimensional embedding
        labels (ndarray): discrete labels to compute the laplacian
        emb_nbrs (sklearn.neighbors.NearestNeighbors): nearest neighbor search
            data structure fitted on the low-dimensional embedding
        K_use (ndarray): neighborhood sizes
        sample (ndarray, optional): Indices of data subsample for faster
            computation. Defaults to None.

    Returns:
        data frame: with columns "laplacian" and "k"
    """
    if sample is None:
        sample = np.arange(emb.shape[0])
    n = len(sample)
    labels = np.asarray(labels)
    print(f"Computing Laplacian scores for {n} of {emb.shape[0]} points...")
    Kmax = np.max(K_use)
    lap = np.zeros((K_use.shape[0],))

    for i in tqdm(sample):
        emb_nn = emb_nbrs.kneighbors(
            emb[i, :].reshape(1, -1), n_neighbors=Kmax + 1, return_distance=False
        )[
            0
        ]  # do not remove the point itself
        diff_label = labels[emb_nn] != labels[i]
        # need to make sure the point self is the 0th position
        assert diff_label[0] == 0
        # K_use will index the correct position as we keept the point self
        lap = np.add(lap, np.cumsum(diff_label)[K_use])
    lap = lap / (n * K_use)
    result = pd.DataFrame({"k": K_use, "laplacian": lap})
    return result


def get_rnx_auc(rnx, ks):
    """Compute the AUC for a RNX curve.

    Args:
        rnx (np array): values for rnx values
        ks (np array): neighborhood sizes for rnx values
    """
    return np.sum(rnx / ks) / np.sum(np.ones_like(ks.shape[0]) / ks)


def rnx_adjusted(emb, orig, labels, emb_nbrs, orig_nbrs, n_neigh, sample=None):
    """Evaluation score assessing high and low-dimensional neighborhood
    overlap. We first find the n_neigh nearest neighbors in the embedding
    and then select equally many high-dimensional neighbors with same and
    different label.

    Args:
        emb (ndarray): low-dimensional embedding
        orig (ndarray): high-dimensional data
        labels (ndarray): discrete labels used to adjust the set of neighbors
        emb_nbrs (sklearn.neighbors.NearestNeighbors): nearest neighbor search
            data structure fitted on the low-dimensional embedding
        orig_nbrs (sklearn.neighbors.NearestNeighbors): nearest neighbor search
            data structure fitted on the high-dimensional embedding
        n_neigh (int): neighborhood size
        sample (ndarray, optional): Indices of data subsample for faster
            computation. Defaults to None.
    Returns:
        float: adjusted RNX score
    """
    if sample is None:
        sample = np.arange(emb.shape[0])
    n = len(sample)
    print(f"Computing R_NX value with {n} of {emb.shape[0]} points...")
    qnx = 0
    rnx = 0

    for i in tqdm(sample):
        k_embnn = emb_nbrs.kneighbors(
            emb[i, :].reshape(1, -1), n_neighbors=n_neigh + 1, return_distance=False
        )[0, 1:]
        k_orignn = orig_nbrs.kneighbors(
            orig[i, :].reshape(1, -1), n_neighbors=emb.shape[0], return_distance=False
        )[0, 1:]
        k_embnn_same = k_embnn[np.nonzero(labels[k_embnn] == labels[i])[0]]
        k_embnn_diff = k_embnn[np.nonzero(labels[k_embnn] != labels[i])[0]]
        num_same = len(k_embnn_same)

        k_orignn_same = k_orignn[np.nonzero(labels[k_orignn] == labels[i])[0]]
        k_orignn_diff = k_orignn[np.nonzero(labels[k_orignn] != labels[i])[0]]

        qnx = (
            qnx
            + len(
                set(k_embnn_same[:num_same]).intersection(set(k_orignn_same[:num_same]))
            )
            + len(
                set(k_embnn_diff[: (n_neigh - num_same)]).intersection(
                    set(k_orignn_diff[: (n_neigh - num_same)])
                )
            )
        )

    rnx = (((emb.shape[0] - 1) * qnx / (n_neigh * n)) - n_neigh) / (
        emb.shape[0] - 1 - n_neigh
    )
    return rnx


def rnxcurves_adjusted(emb, orig, labels, emb_nbrs, orig_nbrs, K_use, sample=None):
    """RNX scores for different neighborhood sizes.

    Args:
        emb (ndarray): low-dimensional embedding
        orig (ndarray): high-dimensional data
        labels (ndarray): discrete labels used to adjust the set of neighbors
        emb_nbrs (sklearn.neighbors.NearestNeighbors): nearest neighbor search
            data structure fitted on the low-dimensional embedding
        orig_nbrs (sklearn.neighbors.NearestNeighbors): nearest neighbor search
            data structure fitted on the high-dimensional embedding
        K_use (ndarray): neighborhood sizes
        sample (ndarray, optional): Indices of data subsample for faster
            computation. Defaults to None.

    Returns:
        data frame: columns "k" and "adjusted_rnx"
    """
    if sample is None:
        sample = np.arange(emb.shape[0])
    n = len(sample)
    print(f"Computing adjusted R_NX curve with {n} of {emb.shape[0]} points...")

    Kmax = np.max(K_use)
    rnx = np.zeros((K_use.shape[0],))
    qnx = np.zeros((K_use.shape[0],))

    for i in tqdm(sample):
        k_embnn = emb_nbrs.kneighbors(
            emb[i, :].reshape(1, -1), n_neighbors=Kmax + 1, return_distance=False
        )[0, 1:]
        k_orignn = orig_nbrs.kneighbors(
            orig[i, :].reshape(1, -1), n_neighbors=emb.shape[0], return_distance=False
        )[0, 1:]
        k_embnn_same = k_embnn[labels[k_embnn] == labels[i]]
        k_embnn_diff = k_embnn[labels[k_embnn] != labels[i]]

        k_orignn_same = k_orignn[labels[k_orignn] == labels[i]]
        k_orignn_diff = k_orignn[labels[k_orignn] != labels[i]]

        label_cumsum = np.cumsum(labels[k_embnn] == labels[i])
        for j, k in enumerate(K_use):
            num_same = label_cumsum[k - 1]
            qnx[j] = (
                qnx[j]
                + len(
                    set(k_embnn_same[:num_same]).intersection(
                        set(k_orignn_same[:num_same])
                    )
                )
                + len(
                    set(k_embnn_diff[: (k - num_same)]).intersection(
                        set(k_orignn_diff[: (k - num_same)])
                    )
                )
            )

    for j, k in enumerate(K_use):
        rnx[j] = (((emb.shape[0] - 1) * qnx[j] / (k * n)) - k) / (emb.shape[0] - 1 - k)
    result = pd.DataFrame({"k": K_use, "adjusted_rnx": rnx})
    return result


def adjusted_jaccard_similarity(
    emb,
    orig,
    labels,
    n_neigh,
    emb_nbrs,
    orig_nbrs,
    sample=None,
):
    """Adjusted Jaccard similarity of high- and low-dimensional 
    fixed-sized neighborhoods. 

    Args:
        emb (ndarray): low-dimensional embedding
        orig (ndarray): high-dimensional data
        labels (ndarray): discrete labels used to adjust the set of neighbors
        n_neigh (int): neighborhood size
        emb_nbrs (sklearn.neighbors.NearestNeighbors): nearest neighbor search
            data structure fitted on the low-dimensional embedding
        orig_nbrs (sklearn.neighbors.NearestNeighbors): nearest neighbor search
            data structure fitted on the high-dimensional embedding
        sample (ndarray, optional): Indices of data subsample for faster
            computation. Defaults to None.

    Returns:
        data frame: column "jaccard_similarity"
    """
    if sample is None:
        sample = np.arange(emb.shape[0])
    n = len(sample)
    print(f"Computing jaccard similarity with {n} of {emb.shape[0]} points...")

    jacc = np.full(emb.shape[0], -1, dtype=float)

    for i in tqdm(sample):
        l = labels[i]
        emb_nbrs_i = emb_nbrs.kneighbors(
            emb[i, :].reshape(1, -1), n_neighbors=n_neigh + 1, return_distance=False
        )[0]
        assert (
            emb_nbrs_i[0] == i
        ), f"emb_nbrs_i for i {i}, with emb {emb[i, :]} is {emb_nbrs_i}"
        emb_nbrs_i = emb_nbrs_i[1:]
        A = set(emb_nbrs_i)

        # how many with same label?
        num_same = np.count_nonzero(labels[emb_nbrs_i] == l)
        num_diff = n_neigh - num_same

        orig_nbrs_i = orig_nbrs.kneighbors(
            orig[i, :].reshape(1, -1), n_neighbors=emb.shape[0], return_distance=False
        )[0]
        assert orig_nbrs_i[0] == i
        orig_nbrs_i = orig_nbrs_i[1:]
        orig_same_nn = orig_nbrs_i[np.nonzero(labels[orig_nbrs_i] == l)[0]][:num_same]
        orig_diff_nn = orig_nbrs_i[np.nonzero(labels[orig_nbrs_i] != l)[0]][:num_diff]

        B = set(np.hstack((orig_same_nn, orig_diff_nn)))
        jacc[i] = len(A.intersection(B)) / len(A.union(B))

    result_df = pd.DataFrame({"jaccard_similarity": jacc})
    return result_df


def evaluate_embedding(
    embedding,
    metadata_df,
    data_dict,
    n_neigh,
    prior_col,
    secondary_col,
    evaluation_measures,
    sampling_frac=None,
    wandb_log=True,
):
    """Computing evaluation scores on the low-dimensional embedding.
    Possible 'evaluation_measures' to select:
        - lisi: Local Inverse Simpson Index 
            --> only possible with R libraries installed (see lisi.R)
        - laplacian: Fraction of neighbors with different label
        - jaccard: adjusted Jaccard similarity between HD and LD space
        - rnx: adjusted neighborhood overlap score normlized wrt random embedding
        - curves: laplacian and rnx over different neighborhood sizes

    Args:
        embedding (ndarray): low-dimensional embedding
        metadata_df (data frame): containing the label values for prior_col 
            and secondary_col
        data_dict (dictionary): original data 'X' and factorized prior class labels 'Y'
        n_neigh (int): neighborhood size
        prior_col (string): column name of labels that have been factore out in the embedding
        secondary_col (string): column name of secondary label structure to be evaluated
        evaluation_measures (list of string): measures to compute
        sampling_frac (float, optional): Fraction to sample a subset of points for evaluation. Defaults to None.
        wandb_log (bool, optional): Logging results to weights & biases. Defaults to True.
    
    Returns:
        data frame with evaluation measures per point
    """
    
    if wandb_log is False:
        wandb.init(mode="disabled")
    
    for m in evaluation_measures:
        assert m in ['lisi', 'laplacian', 'jaccard', 'rnx', 'curves'], f"Evaluation measure {m} not known."

    pointwise_results = pd.DataFrame()
    if sampling_frac is not None:
        sample = np.random.choice(
            embedding.shape[0],
            size=int(sampling_frac * embedding.shape[0]),
        )
        print(
            f"Evaluating embedding on {len(sample)} out of {embedding.shape[0]} points."
        )
    else:
        sample = None

    # Local Inverse Simpsons Index
    if "lisi" in evaluation_measures:
        lisi = compute_lisi(
            metadata_df[[prior_col, secondary_col]].join(pd.DataFrame({'x': embedding[:, 0],
                                                                       'y': embedding[:, 1]})),
            perplexity=n_neigh,
        )
        wandb.log(
            {
                lisi.columns[0] + "_median": lisi[lisi.columns[0]].median(),
                lisi.columns[1] + "_median": lisi[lisi.columns[1]].median(),
                lisi.columns[2] + "_median": lisi[lisi.columns[2]].median(),
                lisi.columns[3] + "_median": lisi[lisi.columns[3]].median(),
            }
        )
        pointwise_results = pd.concat([pointwise_results, lisi])

    # Build nearest neighbor structure for evaluations
    if any(m in evaluation_measures for m in ["laplacian", "jaccard", "rnx", "curves"]):
        print("Building nearest neighbor search datastructure...", end="")
        emb_nbrs = NearestNeighbors(p=2).fit(embedding)
        orig_nbrs = NearestNeighbors(p=2).fit(data_dict["X"])
        print("Done.")

    # Laplacian Score
    if "laplacian" in evaluation_measures:
        lap = compute_laplacian(
            embedding,
            data_dict["Y"],
            n_neigh=n_neigh,
            emb_nbrs=emb_nbrs,
            sample=sample,
        )
        random_lap = get_random_laplacian(
            num_points=embedding.shape[0], labels=data_dict["Y"]
        )
        wandb.log({"laplacian_mean": np.mean(lap[lap != -1])})
        wandb.log({"laplacian_median": np.median(lap[lap != -1])})
        wandb.log(
            {
                "laplacian_"
                + ("" if prior_col is None else prior_col)
                + "_expected": random_lap
            }
        )
        pointwise_results["laplacian"] = lap

        if secondary_col is not None:
            lap_shape = compute_laplacian(
                embedding,
                metadata_df[secondary_col],
                n_neigh=n_neigh,
                emb_nbrs=emb_nbrs,
                sample=sample,
            )
            random_lap_shape = get_random_laplacian(
                num_points=embedding.shape[0], labels=metadata_df[secondary_col]
            )
            wandb.log(
                {
                    "laplacian_"
                    + secondary_col
                    + "_mean": np.mean(lap_shape[lap_shape != -1])
                }
            )
            wandb.log(
                {
                    "laplacian_"
                    + secondary_col
                    + "_median": np.median(lap_shape[lap_shape != -1])
                }
            )
            wandb.log({"laplacian_" + secondary_col + "_expected": random_lap_shape})
            pointwise_results["laplacian_" + secondary_col] = lap_shape

    # Jaccard index
    if "jaccard" in evaluation_measures:
        jacc_df = adjusted_jaccard_similarity(
            embedding,
            data_dict["X"],
            n_neigh=n_neigh,
            labels=data_dict["Y"],
            emb_nbrs=emb_nbrs,
            orig_nbrs=orig_nbrs,
            sample=sample,
        )
        wandb.log(
            {
                "jaccard_similarity_median": np.median(
                    jacc_df.loc[
                        jacc_df["jaccard_similarity"] != -1, "jaccard_similarity"
                    ]
                )
            }
        )
        wandb.log(
            {
                "jaccard_similarity_mean": np.mean(
                    jacc_df.loc[
                        jacc_df["jaccard_similarity"] != -1, "jaccard_similarity"
                    ]
                )
            }
        )
        pointwise_results = pointwise_results.join(jacc_df)

    if "rnx" in evaluation_measures:
        rnx = rnx_adjusted(
            embedding,
            data_dict["X"],
            data_dict["Y"],
            emb_nbrs=emb_nbrs,
            orig_nbrs=orig_nbrs,
            n_neigh=n_neigh,
            sample=sample,
        )
        wandb.log({"adjusted_RNX": rnx})

    if "curves" in evaluation_measures:
        result_curves = pd.DataFrame()
        # which neighborhood sizes to evaluate?
        K_use = np.logspace(
            start=0, stop=np.log10(embedding.shape[0] - 2), base=10, num=15
        )
        K_use = np.unique(np.asarray(np.round(K_use), dtype=int))
        K_use = K_use[: len(K_use) - 1]
        result_curves["k"] = K_use
        print(f"\nComputing evaluation measures on k in \n{list(K_use)}")

        # Laplacian
        laplacian_curve_df = laplacian_curve(
            embedding,
            labels=data_dict["Y"],
            emb_nbrs=emb_nbrs,
            K_use=K_use,
            sample=sample,
        )
        result_curves = result_curves.join(laplacian_curve_df.set_index("k"), on="k")

        if secondary_col is not None:
            laplacian_group_curve_df = laplacian_curve(
                embedding,
                labels=metadata_df[secondary_col],
                emb_nbrs=emb_nbrs,
                K_use=K_use,
                sample=sample,
            )
            laplacian_group_curve_df.rename(
                {"laplacian": "laplacian_" + secondary_col}, axis=1, inplace=True
            )
            result_curves = result_curves.join(
                laplacian_group_curve_df.set_index("k"), on="k"
            )

        # RNX
        arnx_curve_df = rnxcurves_adjusted(
            embedding,
            data_dict["X"],
            labels=data_dict["Y"],
            emb_nbrs=emb_nbrs,
            orig_nbrs=orig_nbrs,
            K_use=K_use,
            sample=sample,
        )
        result_curves = result_curves.join(arnx_curve_df.set_index("k"), on="k")
        result_curves.loc[result_curves.shape[0]] = {
            "k": embedding.shape[0] - 1,
            "laplacian": random_lap,
            "laplacian_" + secondary_col: random_lap_shape,
            "adjusted_rnx": 0,
        }
        wandb.log({"evaluation_curves": wandb.Table(dataframe=result_curves)})

    pointwise_results["k"] = n_neigh
    pointwise_results = metadata_df.join(pointwise_results)
    wandb.log({"result_table": wandb.Table(dataframe=pointwise_results)})
    return pointwise_results