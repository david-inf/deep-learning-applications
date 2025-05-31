"""CLI arguments for main programs"""

from types import SimpleNamespace
import argparse
import yaml
from utils import LOG


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    help="YAML configuration file")
parser.add_argument("--view", action="store_true",  # default False
                    help="Visualize model architecture, no training")


def print_info(opts):
    LOG.info("Training for num_epochs=%s", opts.num_epochs)


def parse_args():
    """Parse arguments given via CLI"""
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)  # return dict
    opts = SimpleNamespace(**configs)

    opts.visualize = args.view
    opts.config_file = args.config
    print_info(opts)

    return opts
