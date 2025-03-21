""" Script for generating YAML configuration files for experiments """

import os
import yaml
from types import SimpleNamespace
from ipdb import launch_ipdb_on_exception
import torch
from torchinfo import summary
from lab1.lab1 import get_model


MODELS = ("MLP", "CNN")

INPUT_SIZE = {
    "MNIST": torch.randn(128, 1, 28, 28),
    "CIFAR10": torch.randn(128, 3, 28, 28)
}


def gen_configs(new_params, base_yaml="config.yaml"):
    # new_params may contain updated and new parameters
    # Load base configuration file
    with open(base_yaml, "r") as f:
        base_config = yaml.safe_load(f)  # dict

    # Update with new parameters
    base_config.update(new_params)

    # Automate naming
    _prefix = ("skip" if base_config["skip"]
               else "") + base_config["model_name"]
    _dataset = base_config["dataset"].lower()
    _opts = SimpleNamespace(**base_config)
    _model_stats = summary(get_model(_opts), verbose=0)
    _params = f"{_model_stats.total_params/10**6:.2f}M"
    base_config["experiment_name"] = f"{_prefix}_{_params}_{_dataset}"

    # Dump new configuration file
    output_dir = "experiments"
    os.makedirs(output_dir, exist_ok=True)

    fname = base_config["experiment_name"] + ".yaml"
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w") as f:
        yaml.dump(base_config, f)

    print(f"Generated config: {output_path}")
    print(f"Parameters: {base_config}")
    print(f"Experiment name: {base_config["experiment_name"]}")


if __name__ == "__main__":
    configs = [  # list of params (dict)
        # {"model_name": "MLP", "dataset": "MNIST", "skip": False,
        #  "n_blocks": 2, "hidden_size": 512, },
        # {"model_name": "MLP", "dataset": "MNIST", "skip": True,
        #  "n_blocks": 2, "hidden_size": 512, },
        # {"model_name": "MLP", "dataset": "CIFAR10", "skip": False,
        #  "n_blocks": 1, "hidden_size": 512, },
        # {"model_name": "MLP", "dataset": "CIFAR10", "skip": True,
        #  "n_blocks": 1, "hidden_size": 512, },
        ## **** ##
        # {"model_name": "CNN", "dataset": "MNIST", "skip": False,
        #  "num_filters": 32, },
        # {"model_name": "CNN", "dataset": "MNIST", "skip": True,
        #  "num_filters": 32, },
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False,
        #  "num_filters": 64, },
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": True,
        #  "num_filters": 64, },

        ## **** ##

        {
            "teacher": {
                "model_name": "CNN",
                "num_filters": 32,
            }
        }
    ]
    for params in configs:
        with launch_ipdb_on_exception():
            gen_configs(params)
            print()
    print("Done")
