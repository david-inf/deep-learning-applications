""" Main script for training a single model """

from comet_ml import start

import random
import numpy as np
import torch

from train import train_loop, test
from utils import (LOG, update_yaml, set_seeds,
                   get_loaders, get_model)


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
    import cmd_args
    from ipdb import launch_ipdb_on_exception
    opts = cmd_args.parse_args()

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
