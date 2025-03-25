""" Arguments for the main programs """

import os
from types import SimpleNamespace
import argparse
import yaml
import torch
from utils import LOG


parser = argparse.ArgumentParser(
    description="Run an experiment and log to comet_ml")

parser.add_argument("--config", help="YAML configuration file")

parser.add_argument("--epochs", default=10, type=int,
                    help="Number of epochs, increase when resuming")
parser.add_argument("--ckping", type=int, default=None,
                    help="Specify checkpointing frequency with epochs")

parser.add_argument("--log_every", type=int, default=20,
                    help="Metrics logging frequency in batches")
# parser.add_argument("--window", type=int, default=60,
#                     help="Number of previous batches when computing metrics")


def update_opts(opts, args):
    # update yaml file with updated and new attributes from opts
    # Configs yaml file
    opts.config = args.config  # keep the yaml file name

    # Device
    opts.device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info(f"Device: {opts.device}")

    # Update epochs
    if args.epochs != opts.num_epochs:
        prev = opts.num_epochs
        opts.num_epochs = args.epochs
        LOG.info(f"Updated number of epochs to {opts.num_epochs} from {prev}")
    else:
        LOG.info(f"Training for {opts.num_epochs} epochs")

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

    # Early stopping
    if hasattr(opts, "early_stopping"):
        patience = opts.early_stopping["patience"]
        threshold = opts.early_stopping["threshold"]
        LOG.info(f"Early stopping activated with patience {patience}, "
                 f"threshold {threshold}")

    # Update opts with new attributes from args
    opts.__dict__.update(vars(args))

    # Update yaml file
    with open(opts.config, "w") as f:
        # dump the updated opts to the yaml file
        yaml.dump(opts.__dict__, f)


def parse_args():
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    opts = SimpleNamespace(**configs)
    update_opts(opts, args)

    return opts
