"""Miscellaneous of utilities"""

import random
import logging
import torch
import numpy as np
from rich.logging import RichHandler
import yaml


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log

LOG = get_logger()


def update_yaml(opts, key, value):
    """
    Update a key in the yaml configuration file, overwriting it

    Args:
        opts (SimpleNamespace): the configuration object
        key (str): the key to update
        value (any): the new value
    """
    # update the opts object
    opts.__dict__[key] = value
    # update the yaml file
    with open(opts.config, "w", encoding="utf-8") as f:
        # dump the updated opts to the yaml file
        yaml.dump(opts.__dict__, f)


def set_seeds(seed):
    """ Set seeds for all random number generators """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def visualize(model, model_name, input_data):
    from torchinfo import summary
    from rich.console import Console
    out = model(input_data)

    console = Console()
    console.print(f"model={model_name}, output_shape={out.shape}")

    model_stats = summary(
        model,
        input_data=input_data,
        col_names=[
            # "input_size",
            "output_size",
            "num_params",
            # "params_percent",
            "kernel_size",
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
