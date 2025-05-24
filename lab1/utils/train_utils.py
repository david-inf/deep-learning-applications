"""Training-specific utilities"""

import os
import torch
import numpy as np
from lab1.utils import LOG, update_yaml


def N(x: torch.Tensor):
    """Get actual value"""
    # detach from computational graph
    # send back to cpu
    # numpy ndarray
    return x.detach().cpu().numpy()


def accuracy(logits, labels):
    """Compute accuracy during pytorch training"""
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == labels)
    return acc


# TODO: rename to save_torch_model
# TODO: outside module with all utilities for all labs
def save_checkpoint(opts, model: torch.nn.Module, fname=None):
    """Save a model checkpoint to be resumed later"""
    if not fname:
        # fname = f"e_{info["epoch"]:03d}_{opts.experiment_name}.pt"
        fname = f"{opts.experiment_name}.pt"
    os.makedirs(opts.checkpoint_dir, exist_ok=True)
    output_path = os.path.join(opts.checkpoint_dir, fname)

    ckpt_state = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(ckpt_state, output_path)

    # Update yaml file with checkpoint name
    update_yaml(opts, "checkpoint", output_path)
    LOG.info("Saved model at path=%s", opts.checkpoint)


def load_checkpoint(ckpt_path: str, model: torch.nn.Module):
    """Load a model checkpoint to resume training"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_path}")

    # load from given checkpoint path
    LOG.info("Loading checkpoint=%s", {ckpt_path})
    # checkpoint = torch.load(checkpoint_path, map_location="cuda")
    checkpoint = torch.load(ckpt_path)

    # load weights and optimizer in those given
    # this means that the initialized model and optimizer are updated
    model.load_state_dict(checkpoint["model_state_dict"])


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        """Initialize the AverageMeter with default values."""
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def reset(self):
        """Reset all statistics to zero."""
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def update(self, val, n=1):
        """Update statistics with new value.

        Args:
            val: The value to update with
            n: Weight of the value (default: 1)
        """
        # update statistic with given new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping strategy, implicit regularization

    Args:
        patience: epochs with no improvement to wait
        min_delta: minimum change for improvement
    """

    def __init__(self, opts, verbose=True):
        self._opts = opts
        self.patience = opts.early_stopping["patience"]
        self.min_delta = opts.early_stopping["min_delta"]
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            # initialize best score
            self.best_score = score
            self.checkpoint(model)  # first model
        elif score < self.best_score + self.min_delta:
            # no improvement seen
            self.counter += 1
            if self.verbose:
                LOG.info("Early stopping counter=%s out of patience=%s",
                         self.counter, self.patience)
            if self.counter >= self.patience:
                # stop training when we see no improvements
                self.early_stop = True
                # at this point we should stop training
                # and save checkpoint
        else:
            # we see an improvement
            self.best_score = score
            self.checkpoint(model)
            self.counter = 0

    def checkpoint(self, model):
        """Save current best model"""
        LOG.info("Updated best_score=%.3f", self.best_score)
        save_checkpoint(self._opts, model)
