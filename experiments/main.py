import wandb
import numpy as np
import argparse
import plot_utils
import yaml
import time
import random
import sys

sys.path.append("../ctsne/")
from bhtsne import run_bh_tsne as ctsne

sys.path.append("../fitsne/")
from fast_ctsne import fast_ctsne as fastctsne

from dataset import get_dataset
import evaluation

# Default parameters
DEFAULT_SEED = 42


def get_method(name):
    if name == "fastctsne":
        return fastctsne
    elif name == "ctsne":
        return ctsne
    elif name == "tsne":
        return tsne_wrapper
    else:
        raise ValueError(f"Method name {name} not known.")


def tsne_wrapper(data, **kwargs):
    return fastctsne(data, alpha=1.0, **kwargs)

def dot_to_dictionary(config):
    dict_config = dict()
    param_keys, param_values = config.keys(), list(config.values())

    # Transform from dot.notation to {"dot": notation}
    for i, key in enumerate(param_keys):
        split_key = str.split(key, sep=".")
        if len(split_key) > 1:
            subkey = split_key[0]
            if subkey not in dict_config:
                dict_config[subkey] = {split_key[1]: param_values[i]}
            else:
                dict_config[subkey][split_key[1]] = param_values[i]
        else:
            dict_config[key] = param_values[i]
    return dict_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to .yaml config file.")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    return dot_to_dictionary(config)


def main(config):
    dataset_name = config["dataset"]
    method_name = config["method"]

    wandb.init(
        project="your-project-name",
        entity="your-username",
        tags=[dataset_name, method_name],
        config=config,
    )

    seed = config.get("seed", DEFAULT_SEED)
    random.seed(seed)
    np.random.seed(seed)
    data_dict = get_dataset(
        config["dataset"],
        prior_col=config.get("prior_col"),
        label_col=config.get("label_col", None),
        **config.get(config["dataset"], {}),
    )
    print(f"Read {config['dataset']} data of size {data_dict['X'].shape}.")

    emb_fn = get_method(method_name)
    start = time.time()
    embedding = emb_fn(
        data_dict["X"],
        labels=data_dict["Y"],
        **config["training"],
        **config[method_name],
    )
    embedding_time = time.time() - start

    metadata_df = data_dict["metadata_df"]
    metadata_df["x"] = embedding[:, 0]
    metadata_df["y"] = embedding[:, 1]

    # Log embedding
    wandb.log({"runtime": embedding_time})

    wandb.log(
        {
            "embedding_img": wandb.Image(
                plot_utils.plot_embedding(
                    metadata_df,
                    color_col=config.get("prior_col", None),
                    shape_col=config.get("label_col", None),
                )
            )
        }
    )
    wandb.log(
        {
            "embedding_plt": plot_utils.plotly_scatter(
                metadata_df,
                color_col=config.get("prior_col", None),
                shape_col=config.get("label_col", None),
            )
        }
    )

    _ = evaluation.evaluate_embedding(
        embedding=embedding,
        metadata_df=metadata_df,
        data_dict=data_dict,
        n_neigh=config.get("n_neigh", config.get("perplexity", 30)),
        prior_col=config["prior_col"],
        secondary_col=config.get("label_col", None),
        evaluation_measures=config.get("evaluation_measures", []),
        sampling_frac=config.get("evaluation_sampling", None),
    )


if __name__ == "__main__":
    exit(main(parse_args()))
