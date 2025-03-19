
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
