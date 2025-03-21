""" Main script for distillation """

import os
from types import SimpleNamespace

from comet_ml import start

import torch

from train import train_loop_distill, test
from utils import (LOG, update_yaml, set_seeds,
                   get_loaders, get_model)


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
    print(teacher)
    print(student)

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
    import cmd_args
    from ipdb import launch_ipdb_on_exception
    opts = cmd_args.parse_args()

    with launch_ipdb_on_exception():
        main(opts)
