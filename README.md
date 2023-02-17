# Revised Conditional t-SNE: Looking Beyond the Nearest Neighbors

Implementation of our paper [Revised Conditional t-SNE](https://arxiv.org/abs/2302.03493) presented at the International Symposium on Intelligent Data Analysis (IDA) 2023.

![Banner](/experiments/synthetic/synthetic_embeddings.png)

## Installation ##

We include the code for ct-SNE and revised ct-SNE (based on FIt-SNE) in this repository. The original repositories are [ct-SNE](https://bitbucket.org/ghentdatascience/ct-sne/src/master/)[1] and [FIt-SNE](https://github.com/KlugerLab/FIt-SNE.git)[2].


### Conda environment ###
Install the required packages e.g. in a new local enviroment such as
```bash
conda env create --prefix ./ctsne_env -f requirements.yaml
conda activate ctsne_env/
```

Note: to preprocess the human immune dataset you also need  [scib](https://github.com/theislab/scib) and scanpy.


The experiments were logged using **Weights & Biases**. If you have your own account you might want to login first and enable synchronization with your account by changing the following lines in ```main.py```.

```python
wandb.init(
project="your-projectname",
entity="your-username",
tags=[dataset_name, method_name],
config=config,
)
```

Running the code without Weights & Biases works fine when you disable the package after activating the conda environment.
```bash
conda activate ctsne_env
wandb disabled
```
You might want to write add functionality to store the results locally.


### Conditional t-SNE ###
Compile from the /ctsne/ directory 
```bash
g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2
```

### Revised Conditional t-SNE based on FIt-SNE ###
Prerequisite: [FFTW](http://www.fftw.org/)

Command to compile 'fast_cstne' (revised ct-SNE) from the ```fitsne``` directory.
```bash
g++ -std=c++11 -O3  src/sptree.cpp src/tsne.cpp src/nbodyfft.cpp  -o bin/fast_ctsne -pthread -lfftw3 -lm -Wno-address-of-packed-member
```


## Datasets ##

The datasets are all read in 'dataset.py' and their path and filenames can be adjusted there. 

### Synthetic Data ###
The synthetic data ```experiments/synthetic/synthetic_data.csv``` has been generated using the characteristics specified in the config files. 

### Pancreas Data ###
We include the processed 50-dimensional data ```experiments/pancreas/pancreas_pca.rds``` and the metadata ```/experiments/pancreas/metadata.rds```.


### Human Immune Data ###
The data aggregated by Luecken et al. [3] can be downloaded [here](https://figshare.com/ndownloader/files/25717328). It contains measurements for n=33506 cells accross d=12303 genes. We store it as ```experiments/immune/Immune_ALL_human.h5ad``` and processed it using the same batch-aware hvg selection. To do so on your own

- Install scanpy and scib (potentially in a different conda environment as scib requires Python >=3.7)
- Process the human immune data using ```experiments/immune/immune_preprocessing.py```.
- Make sure the filepaths in ```dataset.py``` point to the processed data.


## Experiments

We provide two jupyter notebooks ```synthetic_example.ipynb``` and ```pancreas_example``` to show how revised ct-SNE can be used and compared against t-SNE and ct-SNE. To perform more extensive experiments the training parameters can be specified using yaml configuration files in ```experiments/config/```.
Providing the path as a command line argument will run the specified method/dataset combination.

```bash
conda activate your-ctsne-env
cd experiments
python main.py config/synthetic_fastctsne.yaml
```

To compute embeddings for e.g. different values of $\beta$, use the W&B sweep configs:
```bash
python wandb_sweep.py config/sythetic_fastctsne_sweep.yaml
``` 

Finally, we provide the tables with all evaluation measures in the respective data subfolders.

## References ##

If our work is helpful to you, please cite our paper as:
```
@inproceedings{heiter2023revised,
  title={Revised Conditional t-SNE: Looking Beyond the Nearest Neighbors},
  author={Heiter, Edith and Kang, Bo and Seurinck, Ruth and Lijffijt, Jefrey},
  booktitle={Advances in Intelligent Data Analysis XXI},
  year={2023},
  publisher="Springer International Publishing",
  address="Cham",
  pages="TBD",
}
```

[1] Kang, B., García García, D., Lijffijt, J., Santos-Rodríguez, R., De Bie, T.: Conditional t-SNE: more informative t-SNE embeddings. Machine Learning (2021)

[2]  Linderman, G.C., Rachh, M., Hoskins, J.G., Steinerberger, S., Kluger, Y.: Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq
data. Nature methods (2019)

[3] Luecken, M.D., Büttner, M., Chaichoompu, K., Danese, A., Interlandi, M.,
Müller, M.F., Strobl, D.C., Zappia, L., Dugas, M., Colomé-Tatché, M., et al.: Benchmarking atlas-level data integration in single-cell genomics. Nature methods (2022)