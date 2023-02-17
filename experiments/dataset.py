import numpy as np
import pandas as pd
import scipy
from itertools import product
import pyreadr
import os


def get_dataset(name, **kwargs):
    if name == "synthetic":
        return generate_synthetic_dataset(**kwargs)
    elif name == "pancreas":
        return get_pancreas_dataset(**kwargs)
    elif name == "immune_hvgbatch":
        return get_immune_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name {name}.")


def get_immune_dataset(prior_col="batch", **kwargs):
    """Load preprocessed human immune dataset.

    Args:
        prior_col (str, optional): Name of column with label to factor out. Defaults to "batch".

    Returns:
        dictionary with elements
        'X': array_like
        'Y': array_like, labels to factor out
        'metadata_df': data frame with metadata used for visualization
    """
    data = np.loadtxt(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "immune",
            "Immune_ALL_human_pca_batchaware.csv",
        ),
        delimiter=",",
    )
    metadata = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "immune",
            "Immune_ALL_human_meta_batchaware.csv",
        ),
        delimiter=",",
    )
    metadata.drop(["index"], axis=1, inplace=True)
    assert (
        prior_col in metadata.columns
    ), f"{prior_col} is not in the metadata columns {metadata.columns}"

    data_dict = {}
    data_dict["X"] = data
    data_dict["Y"] = np.asarray(pd.factorize(metadata[prior_col])[0])
    data_dict["metadata_df"] = metadata
    return data_dict


def get_pancreas_dataset(prior_col="tech", label_col="celltype", **kwargs):
    """Loading the preprocessed pancreas dataset.

    Args:
        prior_col (str, optional): Name of column with label to factor out. Defaults to "tech".
        label_col (str, optional): Name of column with labels to 
            use for visualization. Defaults to "celltype".

    Returns:
        dictionary with elements
        'X': array_like
        'Y': array_like, labels to factor out
        'metadata_df': data frame with metadata used for visualization
    """
    meta_path = "./pancreas/metadata.rds"
    data_path = "./pancreas/pancreas_pca.rds"

    # metadata
    metadata = pyreadr.read_r(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), meta_path)
    )
    metadata = metadata[None].reset_index()
    metadata.rename(columns={"rownames": "Cell"}, inplace=True)
    assert (
        prior_col in metadata.columns
    ), f"{prior_col} is not in the metadata columns {metadata.columns}"
    assert (
        label_col in metadata.columns
    ), f"{label_col} is not in the metadata columns {metadata.columns}"

    data_pca = pyreadr.read_r(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
    )[None]
    data_pca.index.name = "Cell"
    assert (data_pca.index.values == metadata["Cell"]).all()

    Y = np.asarray(pd.factorize(metadata[prior_col])[0])
    X = np.asarray(data_pca)
    data_dict = {}
    data_dict["X"] = X
    data_dict["Y"] = Y
    data_dict["metadata_df"] = metadata[[prior_col, label_col]]
    return data_dict


def sample_point(centroid1, centroid2, noise_dim):
    point = []
    point.extend(centroid1 + np.random.randn(len(centroid1)) * 0.1)
    point.extend(centroid2 + np.random.randn(len(centroid2)) * 0.1)
    point.extend(np.random.randn(noise_dim))
    return np.array(point)


def generate_synthetic_dataset(
    num_c1=2,
    dim_c1=4,
    var_c1=5.0,
    num_c2=2,
    dim_c2=5,
    var_c2=1.0,
    noise_dim=4,
    minimum_points=200,
    c1_sizes=[0.4, 0.6],
    **kwargs,
):
    """Generating a synthetic dataset where each point belongs to two clusters.

    Args:
        num_c1 (int, optional): Number of primary clusters. Defaults to 2.
        dim_c1 (int, optional): Number of dimensions for c1 clusters. Defaults to 4.
        var_c1 (float, optional): Variance for sampling c1 cluster centers. Defaults to 5.
        num_c2 (int, optional): Number of seconday clusters. Defaults to 3.
        dim_c2 (int, optional): Number of dimensions for c2 clusters. Defaults to 2.
        var_c2 (int, optional): Variance for sampling c2 cluster centers. Defaults to 1.
        noise_dim (int, optional): Number of noise dimensions. Defaults to 4.
        minimum_points (int, optional): Size of smallest cluster. Defaults to 200.
        c1_sizes (list, optional): Relative sizes of c1 clusters. Defaults to [0.4, 0.6].

    Returns:
        dictionary with elements
        'X': array_like
        'Y': array_like, labels to factor out
        'metadata_df': data frame with metadata used for visualization
    """

    col_names = ["batch", "celltype"]
    ndim = dim_c1 + dim_c2 + noise_dim

    for i in range(ndim):
        col_names.append("d_" + str(i))
    df_result = pd.DataFrame([], columns=col_names)

    # generate cluster center
    centroids_1 = np.random.randn(num_c1, dim_c1) * var_c1
    centroids_2 = np.random.randn(num_c2, dim_c2) * var_c2

    if c1_sizes is not None:
        num_points = np.zeros(shape=(num_c1, num_c2), dtype=int)

        total_points = (num_c2 * minimum_points) / min(c1_sizes)
        for i, j in product(range(num_c1), range(num_c2)):
            num_points[i, j] = int((total_points * c1_sizes[i]) / num_c2)

    for i, j in product(range(num_c1), range(num_c2)):
        if c1_sizes is None:
            n = minimum_points
        else:
            n = num_points[i, j]
        data = [
            sample_point(centroids_1[i, :], centroids_2[j, :], noise_dim=noise_dim)
            for _ in range(n)
        ]
        data = np.array(data)
        data = np.hstack(
            (np.repeat(i, n)[:, np.newaxis], np.repeat(j, n)[:, np.newaxis], data)
        )
        df_result = pd.concat(
            (df_result, pd.DataFrame(data, columns=col_names)), ignore_index=True
        )

    df_result.iloc[:, 2:] = scipy.stats.zscore(df_result.iloc[:, 2:].values)

    # label combinations
    labels, _ = pd.factorize(
        df_result["batch"].astype(str) + df_result["celltype"].astype(str)
    )
    df_result.insert(2, "combined_cluster", labels)
    df_result["batch"] = df_result["batch"].astype(np.int32)
    df_result["celltype"] = df_result["celltype"].astype(np.int32)
    df_result["combined_cluster"] = df_result["combined_cluster"].astype(np.int32)
    df_result.to_csv("synthetic/synthetic_dataset.csv", sep=',', header=True, index=False)

    data_dict = {}
    data_dict["X"] = np.asarray(df_result.iloc[:, 3:])
    data_dict["Y"] = np.asarray(pd.factorize(df_result["batch"])[0])
    data_dict["metadata_df"] = df_result[["batch", "celltype", "combined_cluster"]]

    return data_dict
