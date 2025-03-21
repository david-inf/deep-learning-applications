""" Script for generating YAML configuration files for experiments """

import os
import yaml
from types import SimpleNamespace
from ipdb import launch_ipdb_on_exception
import torch
from torchinfo import summary
from lab1 import get_model


MODELS = ("MLP", "CNN")

INPUT_SIZE = {
    "MNIST": torch.randn(128, 1, 28, 28),
    "CIFAR10": torch.randn(128, 3, 28, 28)
}


def count_params(configs):
    # returns params in millions
    opts = SimpleNamespace(**configs)
    model_stats = summary(get_model(opts), verbose=0)
    return f"{model_stats.total_params/10**6:.2f}M"


def gen_configs(new_params):
    # new_params may contain updated and new parameters
    # Load base configuration file
    if new_params["model_name"] == "Distill":
        with open("config-distill.yaml", "r") as f:
            base_config = yaml.safe_load(f)  # dict
    else:
        with open("config.yaml", "r") as f:
            base_config = yaml.safe_load(f)  # dict

    # Update with new parameters
    base_config.update(new_params)

    # Automate naming
    _dataset = base_config["dataset"].lower()
    if base_config["model_name"] == "Distill":
        output_dir = "experiments/distill"
        os.makedirs(output_dir, exist_ok=True)
        _teacher = base_config["teacher"]
        _teacher_prefix = _teacher["model_name"] + count_params(_teacher)
        _student = base_config["student"]
        _student_prefix = _student["model_name"] + count_params(_student)
        _prefix = "Distill_" + _teacher_prefix + \
            "_" + _student_prefix
    else:
        output_dir = "experiments/train"
        os.makedirs(output_dir, exist_ok=True)
        _prefix = base_config["model_name"] + ("skip" if base_config["skip"]
                                               else "")
        _params = count_params(base_config)
        _prefix += _params
    base_config["experiment_name"] = f"{_prefix}_{_dataset}"

    # Dump new configuration file
    fname = base_config["experiment_name"] + ".yaml"
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w") as f:
        yaml.dump(base_config, f)

    print(f"Generated config: {output_path}")
    print(f"Configuration: {base_config}")
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
        #  "num_filters": 32, "n_blocks": 4, },
        # {"model_name": "CNN", "dataset": "MNIST", "skip": True,
        #  "num_filters": 32, "n_blocks": 4, },
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False,
        #  "num_filters": 64, "n_blocks": 4},
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": True,
        #  "num_filters": 64, "n_blocks": 4},

        ## **** ##

        {"dataset": "CIFAR10", "model_name": "Distill",
         "teacher": {
             "model_name": "CNN",
             "dataset": "CIFAR10",
             "num_filters": 64,
             "n_blocks": 4,
             "skip": True,
             "ckp": "checkpoints/CNN/e_020_CNNskip_4.83M_cifar10.pt",
             "device": "cuda",
         },
         "student": {
             "model_name": "CNN",
             "dataset": "CIFAR10",
             "num_filters": 16,
             "n_blocks": 2,
             "skip": True,
             "device": "cuda",
         }
         }
    ]

    for params in configs:
        with launch_ipdb_on_exception():
            gen_configs(params)
            print()
    print("Done")
