from main import main as src_main
from main import dot_to_dictionary
import wandb
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to .yaml config file.")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    return dot_to_dictionary(config)


def run_model():
    with wandb.init():
        config = dict(wandb.config)
        dataset_name = config["dataset"]
        method_name = config["method"]
        run = wandb.run
        run.tags = list(run.tags) + [method_name, dataset_name]
        src_main(config)


def main(args):
    config = args
    sweep_id = wandb.sweep(config, project="your-projectname", entity="your-username")
    if config["method"] == "grid":
        count = None
    else:
        count = 20

    wandb.agent(sweep_id, run_model, count=count, project="ctsne")
    print(f"Sweep_id: {sweep_id}")


if __name__ == "__main__":
    main(parse_args())
