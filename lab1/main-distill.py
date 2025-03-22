""" Main script for distillation """

import os
from types import SimpleNamespace

from comet_ml import start

import torch

from train import test
from distill import train_loop_distill
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
    # opts.teacher and opts.student are dict object
    # Teacher must be loaded from a checkpoint
    teacher_opts = SimpleNamespace(**opts.teacher)
    teacher = get_teacher(teacher_opts)

    # Student must be initialized
    student_opts = SimpleNamespace(**opts.student)
    student = get_model(student_opts)

    return student, teacher


def main(opts, experiment):
    # Seed
    set_seeds(opts.seed)
    # Data loaders
    train_loader, val_loader, test_loader = get_loaders(opts)
    # Teacher & Student
    student, teacher = get_teacher_student(opts)
    # Training
    s_opts = SimpleNamespace(**opts.student)
    ckp = s_opts.resume_checkpoint if hasattr(s_opts, "resume_checkpoint") else None
    with experiment.train():
        LOG.info(f"Running {opts.experiment_name}")
        train_loop_distill(opts, teacher, student, train_loader,
                           val_loader, experiment, ckp)
    # Testing
    with experiment.test():
        _, test_acc = test(opts, student, test_loader)
        experiment.log_metrics({"acc": test_acc})
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
