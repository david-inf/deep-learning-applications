""" Main script for distillation """

import os
from types import SimpleNamespace

from comet_ml import start
import torch
from torch.backends import cudnn

from utils.misc_utils import LOG, set_seeds
from main_train import get_loaders, get_model, get_optimization
from distill import train_loop_distill


def get_teacher(teacher_opts):
    """Get teacher model"""
    # Load model
    if not os.path.isfile(teacher_opts.ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_opts.ckpt}")
    LOG.info("Loading teacher from checkpoint=%s", teacher_opts.ckpt)
    ckpt = torch.load(teacher_opts.ckpt)

    model = get_model(teacher_opts)  # already on device
    model.load_state_dict(ckpt["model_state_dict"])
    # Freeze teacher
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model


def get_teacher_student(student_opts, teacher_opts):
    """Get teacher checkpoint and studen model to train"""
    # Teacher must be loaded from a checkpoint
    teacher = get_teacher(teacher_opts)
    # Student must be initialized
    student = get_model(student_opts)
    return student, teacher


def main(opts, experiment):
    """Knowledge distillation training"""
    # Seed
    set_seeds(opts.seed)
    # Data loaders
    train_loader, val_loader = get_loaders(opts)

    # Teacher & Student and Optimizer & Scheduler
    student_opts = SimpleNamespace(**opts.student)
    teacher_opts = SimpleNamespace(**opts.teacher)
    student, teacher = get_teacher_student(student_opts, teacher_opts)
    optimizer, scheduler = get_optimization(student_opts, student)

    # Training
    cudnn.benchmark = True
    with experiment.train():
        LOG.info("Running experiment_name=%s", opts.experiment_name)
        train_loop_distill(opts, teacher, student, optimizer, scheduler,
                           train_loader, val_loader, experiment)
    # TODO: testing? again on validation?


if __name__ == "__main__":
    import cmd_args
    opts = cmd_args.parse_args()

    try:
        experiment = start(project_name=opts.comet_project)
        experiment.set_name(opts.experiment_name)
        experiment.log_parameters(vars(opts))
        main(opts, experiment)
        experiment.log_parameters(vars(opts))
        experiment.end()
    except Exception:
        import ipdb, traceback, sys
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
