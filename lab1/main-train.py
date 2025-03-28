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
    ckp = opts.resume_checkpoint if hasattr(opts, "resume_checkpoint") else None
    with experiment.train():
        LOG.info(f"Running {opts.experiment_name}")
        train_loop(opts, model, train_loader, val_loader,
                   experiment, ckp)
    # Testing
    with experiment.test():
        _, test_acc = test(opts, model, test_loader)
        experiment.log_metric("acc", test_acc)
        LOG.info(f"Test accuracy: {100.*test_acc:.1f}%")


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
