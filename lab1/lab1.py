
import os

from comet_ml import start

import random
import numpy as np
import torch

from mydata import MyMNIST, MyCIFAR10, MakeDataLoaders
from models.mlp import MLP
from models.cnn import CNN

from train import train_loop, test
from utils import LOG, update_yaml


def set_seeds(seed):
    """ Set seeds for all random number generators """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loaders(opts):
    if opts.dataset.lower() == "mnist":
        data = MyMNIST(opts)
    elif opts.dataset.lower() == "cifar10":
        data = MyCIFAR10(opts)
    else:
        raise ValueError(f"Unknown dataset: {opts.dataset}")
    loaders = MakeDataLoaders(opts, data)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    test_loader = loaders.test_loader
    return train_loader, val_loader, test_loader


def get_model(opts):
    # Input size
    if opts.dataset.lower() == "mnist":
        in_channels = 1
        input_size = 28*28 * in_channels
    elif opts.dataset.lower() == "cifar10":
        in_channels = 3
        input_size = 28*28 * in_channels

    # Skip connections
    skip = opts.skip if hasattr(opts, "skip") else False

    if opts.model_name == "MLP":
        blocks = opts.n_blocks if hasattr(opts, "n_blocks") else 2
        hidden = opts.hidden_size if hasattr(opts, "hidden_size") else 512
        model = MLP(input_size, n_blocks=blocks,
                    hidden_size=hidden, skip=opts.skip)
    elif opts.model_name == "CNN":
        filters = opts.num_filters if hasattr(opts, "num_filters") else 64
        model = CNN(in_channels, num_filters=filters, skip=skip)
    else:
        raise ValueError(f"Unknown model: {opts.model_name}")

    model = model.to(opts.device)
    return model


def update_opts(opts, args):
    # update yaml file with updated and new attributes from opts
    # Configs yaml file
    opts.config = args.config  # keep the yaml file name

    # Device
    opts.device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info(f"Device: {opts.device}")

    # Update epochs
    if args.epochs > opts.num_epochs:
        prev = opts.num_epochs
        opts.num_epochs = args.epochs
        LOG.info(f"Updated number of epochs to {opts.num_epochs} from {prev}")

    # Model checkpointing
    if args.ckping:
        opts.checkpoint_every = args.ckping
        LOG.info(f"Checkpointing every {opts.checkpoint_every} epochs")
    else:
        opts.checkpoint_every = opts.num_epochs
        LOG.info(f"Checkpointing at the end of training")
    # checkpoints directory
    ckp_dir = os.path.join("checkpoints", opts.model_name)
    os.makedirs(ckp_dir, exist_ok=True)  # output dir not tracked by git
    opts.checkpoint_dir = ckp_dir  # for saving and loading ckps

    # Update yaml file
    with open(opts.config, "w") as f:
        # dump the updated opts to the yaml file
        yaml.dump(opts.__dict__, f)


def main(opts, experiment):
    # opts: SimpleNmamespace object
    # experiment: comet_ml.Experiment object
    set_seeds(opts.seed)
    # Data loaders
    train_loader, val_loader, test_loader = get_loaders(opts)
    # Model
    model = get_model(opts)
    # Training
    with experiment.train():
        LOG.info(f"Running {opts.experiment_name}")
        train_loop(opts, model, train_loader, val_loader,
                   experiment, opts.resume_checkpoint)
    # Testing
    with experiment.test():
        test_acc = test(opts, model, test_loader)
        LOG.info(f"Test accuracy: {100.*test_acc:.1f}%")
        experiment.log_metrics({"acc": test_acc, "error": 1. - test_acc})


if __name__ == "__main__":
    from types import SimpleNamespace
    from ipdb import launch_ipdb_on_exception
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Run an experiment and log to comet_ml")
    parser.add_argument("--config", default="config.yaml",
                        help="YAML configuration file")
    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of epochs, increase when resuming")
    parser.add_argument("--ckping", type=int, default=None,
                        help="Specify checkpointing frequency with epochs")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)  # dict
    opts = SimpleNamespace(**configs)
    update_opts(opts, args)

    with launch_ipdb_on_exception():
        # try resuming an experiment if experiment_key is provided
        # otherwise start a new experiment
        if not opts.experiment_key:
            experiment = start(project_name=opts.comet_project)
            experiment.set_name(opts.experiment_name)
            # Update with experiment key for resuming
            update_yaml(opts, "experiment_key", experiment.get_key())
            LOG.info("Added experiment key for resuming")
        else:
            # Resume using provided experiment key and checkpoint
            # the key is set above
            # the checkpoint is set with save_checkpoint in train_loop()
            experiment = start(project_name=opts.comet_project,
                               mode="get", experiment_key=opts.experiment_key,)
        main(opts, experiment)
        experiment.log_parameters(vars(opts))
        experiment.end()
