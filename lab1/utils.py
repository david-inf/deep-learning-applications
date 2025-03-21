
def N(x):
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
    from mydata import MyMNIST, MyCIFAR10, MakeDataLoaders
    if opts.dataset.lower() == "mnist":
        data = MyMNIST(opts)
    elif opts.dataset.lower() == "cifar10":
        data = MyCIFAR10(opts)
    else:
        raise ValueError(f"Unknown dataset: {opts.dataset}")
    loaders = MakeDataLoaders(opts, data)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    test_loader = loaders.test_loader
    return train_loader, val_loader, test_loader


def get_model(opts):
    from models.mlp import build_mlp
    from models.cnn import build_cnn

    if opts.model_name == "MLP":
        model, _ = build_mlp(opts)
    elif opts.model_name == "CNN":
        model, _ = build_cnn(opts)
    else:
        raise ValueError(f"Unknown model: {opts.model_name}")

    model = model.to(opts.device)
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
