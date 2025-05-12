
import os
import numpy as np
from transformers import PreTrainedModel
# from accelerate import Accelerator
from utils.misc_utils import LOG, update_yaml


def accuracy(logits, labels):
    """Compute accuracy during pytorch training"""
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == labels)
    return acc


def save_model(opts, model: PreTrainedModel, fname=None):
    """Save a pretrained model"""
    if not fname:
        # fname = f"e_{reached_epoch:02d}_{opts.experiment_name}"
        fname = opts.experiment_name
    os.makedirs(opts.checkpoint_dir, exist_ok=True)
    output_path = os.path.join(opts.checkpoint_dir, fname)

    model.save_pretrained(output_path)
    # add checkpoint path
    update_yaml(opts, "checkpoint", output_path)
    LOG.info("Saved model at path=%s", opts.checkpoint)


def load_model(ckpt_path, model: PreTrainedModel):
    """Load a finetuned model"""
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_path}")

    LOG.info("Loading checkpoint=%s", ckpt_path)
    # TODO: check if it's correct
    model.from_pretrained(ckpt_path, num_labels=2)


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

    def __call__(self, val_acc, model: PreTrainedModel):
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

    def checkpoint(self, model: PreTrainedModel):
        """Save current best model"""
        LOG.info("Updated best_score=%.3f", self.best_score)
        save_model(self._opts, model)

    # def checkpoint(self, accelerator: Accelerator):
    # TODO: make work for multi_gpu
    #     """Current best accelerator state"""
    #     accelerator.save_state()
    #     LOG.info("Saved Accelerator state at path=%s", self._opts.checkpoint)
