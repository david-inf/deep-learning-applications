"""Miscellaneous of utilities"""

import logging
import random
import torch
import numpy as np
from rich.logging import RichHandler


def N(x: torch.Tensor):
    """Get pure value"""
    # detach from computational graph
    # send back to cpu
    # numpy ndarray
    return x.detach().cpu().numpy()


def get_logger():
    """Logging mechanism"""
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log

LOG = get_logger()


def set_seeds(seed):
    """Set seeds for all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    with open(opts.config_file, "w", encoding="utf-8") as f:
        # dump the updated opts to the yaml file
        yaml.dump(opts.__dict__, f)


def visualize(model, model_name, input_data):
    """Model inspection"""
    from torchinfo import summary
    from rich.console import Console

    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    out = model(input_ids=input_ids, attention_mask=attention_mask)

    console = Console()
    console.print(f"Model model={model_name}, computed output_shape={out.logits.shape}")

    model_stats = summary(
        model,
        input_data=input_data,
        col_names=[
            # "input_size",
            "output_size",
            "num_params",
            "params_percent",
            # "mult_adds",
            "trainable",
        ],
        row_settings=("var_names",),
        col_width=18,
        depth=8,
        verbose=0,
    )
    console.print(model_stats)
