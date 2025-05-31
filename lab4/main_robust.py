"""Train a given CNN with the adversarial-augmented dataset"""

import sys
import os
import random

from torch.utils.data import Subset
from torch.backends import cudnn

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from lab1.utils import LOG, set_seeds
from lab1.utils.misc import visualize
from lab1.main_train import get_model, get_optimization

from lab4.mydata import MyCIFAR10, make_loader
from lab4.train_cnn import train_loop


def get_loaders(opts):
    """Get train-val loaders"""
    trainset = MyCIFAR10(opts, train=True)

    # taking a subset allows to do early stopping
    # since the OOD detection will be on the test split
    N = len(trainset)
    indices = list(range(N))
    random.shuffle(indices)
    val_size = int(0.2 * N)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_loader = make_loader(opts, Subset(trainset, train_indices))
    val_loader = make_loader(opts, Subset(trainset, val_indices))

    return train_loader, val_loader


def main(opts):
    """Train a deeep model"""
    # Get loaders
    train_loader, val_loader = get_loaders(opts)
    # Get model, optimizer and scheduler
    model = get_model(opts)
    optimizer, scheduler = get_optimization(opts, model)
    # Launch training
    cudnn.benchmark = True
    LOG.info("experiment_name=%s", opts.experiment_name)
    train_loop(opts, model, optimizer, scheduler, train_loader, val_loader)


def view_model(opts):
    """Model inspection"""
    opts.device = "cpu"
    model, input_data = get_model(opts, True)
    visualize(model, f"{opts.model} on {opts.dataset}", input_data)


if __name__ == "__main__":
    import argparse
    import yaml
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(
        description="Train a CNN with adversarial-augmented dataset")
    parser.add_argument("--config", help="CNN YAML configuration file")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for training (default: cuda:0)")
    # TODO: add arguments for adversarial attacks
    parser.add_argument("--fraction", type=float, default=0.2,
                        help="Fraction of samples to attack (default: 0.2)")
    parser.add_argument("--budget", type=int, default=1,
                        help="Budget for the attack (default: 1)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)  # dict
    opts = SimpleNamespace(**configs)
    opts.fraction = args.fraction
    opts.budget = args.budget

    try:
        if not args.view:
            set_seeds(opts.seed)
            main(opts)
        else:
            view_model(opts)
    except Exception:
        import ipdb, traceback
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
