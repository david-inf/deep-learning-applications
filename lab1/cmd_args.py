""" Arguments for the main programs """

from types import SimpleNamespace
import argparse
import yaml
import torch
from utils import LOG


parser = argparse.ArgumentParser(
    description="Run an experiment and log to comet_ml")

parser.add_argument("--config",
                    # default="lab1/configs/",
                    help="YAML configuration file")
parser.add_argument("--view", action="store_true",  # default False
                    help="Visualize model architecture, no training")


def print_info(opts):
    """Print few training informations"""
    # Device
    # opts.device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("device=%s", opts.device)
    # Epochs
    LOG.info("num_epochs=%d epochs", opts.num_epochs)
    # Model checkpointing
    LOG.info("checkpoint_every=%d epochs", opts.checkpoint_every)

    # Early stopping
    if opts.do_early_stopping:
        patience = opts.early_stopping["patience"]
        min_delta = opts.early_stopping["min_delta"]
        LOG.info("Early stopping activated with patience=%d, "
                 "min_delta=%.4f", patience, min_delta)


def parse_args():
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)  # dict
    opts = SimpleNamespace(**configs)

    opts.visualize = args.view
    opts.config = args.config
    print_info(opts)

    return opts
