
from comet_ml import start, ExperimentConfig

import random
import numpy as np
import torch

from mnist import MyMNIST, MakeDataLoaders
from models import MLP


def set_seeds(seed):
    """ Set seeds for all random number generators """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loaders(opts):
    data = MyMNIST(opts)
    mnist = MakeDataLoaders(opts, data)
    train_loader = mnist.train_loader
    val_loader = mnist.val_loader
    test_loader = mnist.test_loader
    return train_loader, val_loader, test_loader


def get_model(opts):
    if opts.model_name == "MLP":
        model = MLP([128])

    model = model.to(opts.device)
    return model


def main(opts, experiment):
    # opts: SimpleNmamespace object
    # experiment: comet_ml.Experiment object
    set_seeds(opts.seed)

    # Data loaders
    train_loader, val_loader, test_loader = get_loaders(opts)

    # Model
    model = get_model(opts)

    # Training
    # Testing


if __name__ == "__main__":
    from types import SimpleNamespace
    from ipdb import launch_ipdb_on_exception
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Run an experiment and log to comet_ml")
    parser.add_argument("--config", default="config.yaml",
                        help="YAML configuration file")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)  # dict
    opts = SimpleNamespace(**configs)

    with launch_ipdb_on_exception():
        # experiment = start()
        main(opts)
