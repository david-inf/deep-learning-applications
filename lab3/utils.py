
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


def set_seeds(seed):
    """ Set seeds for all random number generators """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def visualize(model, model_name, input_data):
    from torchinfo import summary
    from rich.console import Console

    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    out = model(input_ids=input_ids, attention_mask=attention_mask)

    console = Console()
    console.print(f"Model model={model_name}, computed output shape={out.logits.shape}")

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
