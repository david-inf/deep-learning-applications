""" Main script for training a single model """

from comet_ml import start
import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.backends import cudnn

from mydata import MyMNIST, MyAugmentedMNIST
from mydata import MyCIFAR10, MyAugmentedCIFAR10, MakeDataLoaders
from models.mlp import build_mlp
from models.cnn import build_cnn
from train import train_loop, test
from utils.misc_utils import LOG, set_seeds


def get_loaders(opts):
    """Get train-val loaders"""
    if opts.dataset.lower() == "mnist":
        if hasattr(opts, "augmentation") and opts.augmentation:
            trainset = MyAugmentedMNIST(opts)
        else:
            trainset = MyMNIST(opts)
        valset = MyMNIST(opts, train=False)
    elif opts.dataset.lower() == "cifar10":
        if hasattr(opts, "augmentation") and opts.augmentation:
            trainset = MyAugmentedCIFAR10(opts)
        else:
            trainset = MyCIFAR10(opts)
        valset = MyCIFAR10(opts, train=False)
    else:
        raise ValueError(f"Unknown dataset: {opts.dataset}")

    loaders = MakeDataLoaders(opts, trainset, valset)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    return train_loader, val_loader


def get_model(opts, return_data=False):
    """Get model to train"""
    if opts.model == "MLP":
        model, input_data = build_mlp(opts)
    elif opts.model in ("CNN", "ResNet"):
        model, input_data = build_cnn(opts)
    else:
        raise ValueError(f"Unknown model: {opts.model}")

    model = model.to(opts.device)
    if return_data:
        return model, input_data
    return model


def get_optimization(opts, model: torch.nn.Module):
    """Get Optimizer and LRScheduler"""
    optimizer = optim.SGD(
        model.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay
    )
    if opts.scheduler["type"] == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=opts.scheduler["gamma"])
    elif opts.scheduler["type"] == "multi-step":
        scheduler = MultiStepLR(
            optimizer, opts.scheduler["steps"], opts.scheduler["gamma"])
    else:
        raise ValueError(f"Unknown scheduler {opts.scheduler}")

    return optimizer, scheduler


def main(opts, experiment):
    # opts: SimpleNmamespace object
    # experiment: comet_ml.Experiment object
    set_seeds(opts.seed)
    # Data loaders
    train_loader, val_loader = get_loaders(opts)

    # Model, Optimizer and LRScheduler
    model = get_model(opts)
    optimizer, scheduler = get_optimization(opts, model)

    # Training
    cudnn.benchmark = True
    with experiment.train():
        LOG.info("experiment_name=%s", opts.experiment_name)
        train_loop(opts, model, optimizer, scheduler,
                   train_loader, val_loader, experiment)
    # TODO: Test (validation again, named test)


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
