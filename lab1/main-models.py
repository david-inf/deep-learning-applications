""" Script for inspecting models """

from models.mlp import build_mlp
from models.cnn import build_cnn
from utils import visualize


def main(opts):
    if opts.model_name == "MLP":
        model, input_data = build_mlp(opts)
    elif opts.model_name == "CNN":
        model, input_data = build_cnn(opts)
    else:
        raise ValueError(f"Unknown model {opts.model_name}")

    visualize(model, f"{opts.model_name} on {opts.dataset}", input_data)

if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    import argparse
    import yaml
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="Inspect a given model")
    parser.add_argument("--config", help="YAML configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
    opts = SimpleNamespace(**configs)

    with launch_ipdb_on_exception():
        main(opts)
