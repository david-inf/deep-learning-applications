
import torch


def N(x: torch.Tensor):
    # detach from computational graph
    # send back to cpu
    # numpy ndarray
    return x.detach().cpu().numpy()


def get_logger():
    import logging
    from rich.logging import RichHandler
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log

LOG = get_logger()


class AverageMeter:
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def update(self, val, n=1):
        # update statistic with given new value
        self.val = val  # like loss
        self.sum += val * n  # loss * batch_size
        self.count += n  # count batch samples
        self.avg = self.sum / self.count  # accounts for different sizes


def update_yaml(opts, key, value):
    """
    Update a key in the yaml configuration file

    Args:
        opts (SimpleNamespace): the configuration object
        key (str): the key to update
        value (any): the new value
    """
    import yaml
    # update the opts object
    opts.__dict__[key] = value
    # update the yaml file
    with open(opts.config, "w") as f:
        # dump the updated opts to the yaml file
        yaml.dump(opts.__dict__, f)


def set_seeds(seed):
    """ Set seeds for all random number generators """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loaders(opts):
    from mydata import MyMNIST, MyAugmentedMNIST
    from mydata import MyCIFAR10, MyAugmentedCIFAR10, MakeDataLoaders

    if opts.dataset.lower() == "mnist":
        if hasattr(opts, "augmentation") and opts.augmentation:
            trainset = MyAugmentedMNIST(opts)
        else:
            trainset = MyMNIST(opts)
        valset = MyMNIST(opts)
        testset = MyMNIST(opts, train=False)
    elif opts.dataset.lower() == "cifar10":
        if hasattr(opts, "augmentation") and opts.augmentation:
            trainset = MyAugmentedCIFAR10(opts)
        else:
            trainset = MyCIFAR10(opts)
        valset = MyCIFAR10(opts)
        testset = MyCIFAR10(opts, train=False)
    else:
        raise ValueError(f"Unknown dataset: {opts.dataset}")

    loaders = MakeDataLoaders(opts, trainset, valset, testset)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    test_loader = loaders.test_loader

    return train_loader, val_loader, test_loader


def get_model(opts, return_data=False):
    from models.mlp import build_mlp
    from models.cnn import build_cnn

    if opts.model_name == "MLP":
        model, input_data = build_mlp(opts)
    elif opts.model_name in ("CNN", "ResNet"):
        model, input_data = build_cnn(opts)
    else:
        raise ValueError(f"Unknown model: {opts.model_name}")

    model = model.to(opts.device)

    if return_data:
        return model, input_data
    else:
        return model


def visualize(model, model_name, input_data):
    from torchinfo import summary
    from rich.console import Console
    out = model(input_data)

    console = Console()
    console.print(f"Model {model_name}, computed output shape = {out.shape}")

    model_stats = summary(
        model,
        input_data=input_data,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            # "params_percent",
            # "kernel_size",
            # "mult_adds",
        ],
        row_settings=("var_names",),
        col_width=18,
        depth=8,
        verbose=0,
    )
    console.print(model_stats)
    return model_stats.total_params


def compute_flops(model, input_data, epochs=100, batches=390):
    from torchinfo import summary
    model_stats = summary(model, input_data=input_data, verbose=0)

    # Parameters
    print(f"Params: {model_stats.total_params/1e6:.2f}M")

    # forward pass over each batch
    forward_flops = 2 * model_stats.total_mult_adds  # floating-point operations
    # forward + backward pass over each batch -> FLOP/batch
    flops_batch = 3 * forward_flops
    print(f"Training cost per batch: {flops_batch/1e12:.4f} TFLOP")
    flops_epoch = flops_batch * batches  # batches per epoch
    print(f"Training cost per epoch: {flops_epoch/1e12:.4f} TFLOP")
    total_flops = epochs * flops_epoch
    print(f"Training cost: {total_flops/1e12:.4f} TFLOP")

    gpu_flops = 7 * 1e12  # TFLOPS (per second)
    efficiency = .2
    print(f"Training time: {total_flops/gpu_flops/efficiency:.2f} seconds")
