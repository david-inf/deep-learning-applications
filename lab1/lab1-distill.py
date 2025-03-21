
import os
from types import SimpleNamespace
import yaml

from comet_ml import start

import torch

from lab1.lab1 import set_seeds, get_loaders, get_model, update_opts
from train import train_loop_distill, test
from utils import LOG, update_yaml


def get_teacher(opts):
    # Load model
    if not os.path.isfile(opts.ckp):
        raise FileNotFoundError(f"Teacher file not found: {opts.ckp}")
    LOG.info(f"Loading teacher from {opts.ckp}")
    ckp = torch.load(opts.ckp)

    model = get_model(opts)
    model.load_state_dict(ckp["model_state_dict"])

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model


def get_teacher_student(opts):
    # Teacher must be loaded from a checkpoint
    teacher_opts = SimpleNamespace(**opts.teacher)  # dict
    teacher = get_teacher(teacher_opts)

    # Student must be initialized
    student_opts = SimpleNamespace(**opts.student)  # dict
    student = get_model(student_opts)

    return student, teacher


def main(opts):
    # Seed and Loaders
    set_seeds(opts.seed)
    train_loader, val_loader, test_loader = get_loaders(opts)
    # Teacher & Student
    teacher, student = get_teacher_student(opts)

    # Training
    # with experiment.train():
    LOG.info(f"Running {opts.experiment_name}")
    train_loop_distill(opts, teacher, student, train_loader, val_loader)

    # Testing
    # with experiment.test():
    test_acc = test(opts, student, test_loader)
    LOG.info(f"Test accuracy: {100.*test_acc:.1f}%")
    # experiment.log_metrics({"acc": test_acc})


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    import argparse

    parser = argparse.ArgumentParser(
        description="Run an experiment and log to comet_ml")
    parser.add_argument("--config", default="config-distill.yaml",
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
        main(opts)
