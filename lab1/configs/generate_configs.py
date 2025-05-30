"""Script for generating YAML configuration files for experiments"""

import os
import yaml
# from types import SimpleNamespace
# from torchinfo import summary
# from utils import get_model


# def count_params(configs: dict):
#     # returns params in millions
#     opts = SimpleNamespace(**configs)
#     model_stats = summary(get_model(opts), verbose=0)
#     return f"{model_stats.total_params/10**6:.2f}M"

def gen_configs_distil(new_params):
    """Generate a configuration file given the params dict"""
    configs = {
        "seed": 42, "dataset": "CIFAR10", "batch_size": 128, "num_workers": 4,
        "augmentation": True, "device": "cuda", "num_epochs": 20, "learning_rate": 0.01,
        "momentum": 0.9, "weight_decay": 5e-4,
        # "scheduler": {"type": "multi-step", "steps": [50, 100], "gamma": 0.1},
        "scheduler": {"type": "exponential", "gamma": 0.95},
        "log_every": 20, "checkpoint": None,
        "do_early_stopping": False,
        "comet_project": "deep-learning-applications",

        "temp": 5, "weight_stloss": 10., "weight_labloss": 0.5,
    }
    configs.update(new_params)

    # Checkpoint directory
    output_dir = "lab1/ckpts/Distil"
    os.makedirs(output_dir, exist_ok=True)
    configs["checkpoint_dir"] = output_dir
    # Chekpointing frequency
    if configs.get("checkpoint_every") is None:
        configs["checkpoint_every"] = configs["num_epochs"]
    
    # Configuration file directory
    fname = configs["experiment_name"] + ".yaml"
    output_dir = "lab1/configs/Distil"
    os.makedirs(output_dir, exist_ok=True)
    # Dump configuration file
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(configs, f)

    print(f"Generated file: {output_path}")
    print(f"Configs: {configs}")
    print(f"Experiment: {configs["experiment_name"]}")


def gen_configs_train(new_params):
    """Generate a configuration file given a params dict"""
    configs = {
        "seed": 42, "dataset": "CIFAR10", "batch_size": 128, "num_workers": 4,
        "augmentation": True, "device": "cuda", "num_epochs": 20, "learning_rate": 0.01,
        "momentum": 0.9, "weight_decay": 5e-4,
        # "scheduler": {"type": "multi-step", "steps": [50, 100], "gamma": 0.1},
        "scheduler": {"type": "exponential", "gamma": 0.95},
        "log_every": 20, "checkpoint": None,
        "do_early_stopping": False,
        "comet_project": "deep-learning-applications",
    }
    configs.update(new_params)

    # Experiment naming
    if configs.get("experiment_name") is None:
        _prefix = configs["model"]  # + count_params(configs)
        exp_name = f"{_prefix}_{configs["dataset"].lower()}"
        configs["experiment_name"] = exp_name

    # Checkpoint directory
    if configs.get("checkpoint_dir") is None:
        output_dir = os.path.join("lab1/ckpts", configs["model"])
        configs["checkpoint_dir"] = output_dir
    os.makedirs(configs["checkpoint_dir"], exist_ok=True)

    # Chekpointing frequency
    if configs.get("checkpoint_every") is None:
        configs["checkpoint_every"] = configs["num_epochs"]

    # Configuration file directory
    fname = configs["experiment_name"] + ".yaml"
    output_dir = os.path.join("lab1/configs", configs["model"])
    os.makedirs(output_dir, exist_ok=True)
    # Dump configuration file
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(configs, f)

    print(f"Generated file: {output_path}")
    print(f"Configs: {configs}")
    print(f"Experiment: {configs["experiment_name"]}")


if __name__ == "__main__":
    NUM_FILTERS = 16
    train_new_configs = [  # list of params (dict)

        # MLP
        # {"model": "MLP", "dataset": "MNIST", "layers": [512, 512, 512],
        #  "scheduler": {"type": "multi-step", "steps": [50, 100], "gamma": 0.1},
        #  "experiment_name": "MLP_mnist"},
        # CNN
        # {"model": "CNN", "skip": False, "dataset": "MNIST", "num_filters": 32,
        #  "num_blocks": 2, "experiment_name": "CNN1_mnist"},
        #  {"model": "CNN", "skip": False, "dataset": "MNIST", "num_filters": 64,
        #  "num_blocks": 2, "experiment_name": "CNN2_mnist"},

        # SmallCNN
        # {"model": "CNN", "skip": False, "num_filters": NUM_FILTERS, "num_blocks": 1,
        #  "experiment_name": "SmallCNN"},
        # {"model": "CNN", "skip": True, "num_filters": NUM_FILTERS, "num_blocks": 1,
        #  "experiment_name": "SmallCNNskip"},
        # # MediumCNN
        # {"model": "CNN", "skip": False, "num_filters": NUM_FILTERS, "num_blocks": 5,
        #  "experiment_name": "MediumCNN"},
        # {"model": "CNN", "skip": True, "num_filters": NUM_FILTERS, "num_blocks": 5,
        #  "experiment_name": "MediumCNNskip"},
        # # LargeCNN
        # {"model": "CNN", "skip": False, "num_filters": NUM_FILTERS, "num_blocks": 7,
        #  "experiment_name": "LargeCNN"},
        # {"model": "CNN", "skip": True, "num_filters": NUM_FILTERS, "num_blocks": 7,
        #  "experiment_name": "LargeCNNskip"},

        # BaseCNN for distillation
        # {"model": "CNN", "skip": True, "num_filters": NUM_FILTERS, "num_blocks": 3,
        #  "experiment_name": "BaseCNN"},

        # ResNet (teacher model)
        # {"model": "ResNet", "num_filters": NUM_FILTERS, "num_blocks": 5, "skip": True,
        #  "num_epochs": 50, "early_stopping": {"patience": 4, "min_delta": 0.002}, "do_early_stopping": True,
        #  "experiment_name": "ResNet32",
        #  },
        # WideResNet
        # {"model": "WideResNet", "num_filters": NUM_FILTERS, "num_blocks": 2, "widen_factor": 4,
        #  "num_epochs": 100, "early_stopping": {"patience": 7, "min_delta": 0.002}, "do_early_stopping": True,
        #  "experiment_name": "WideResNet14-4",
        #  },

    ]

    distil_new_configs = [

        # {"dataset": "CIFAR10",
        #  "teacher": {
        #      "model": "ResNet", "dataset": "CIFAR10", "num_filters": NUM_FILTERS, "num_blocks": 9,
        #      "skip": True, "ckpt": "lab1/ckpts/ResNet/ResNet56.pt", "device": "cuda",
        #  },
        #  "student": {
        #      "model": "CNN", "dataset": "CIFAR10", "num_filters": 32, "num_blocks": 1,
        #      "skip": True, "device": "cuda", "weight_decay": 5e-4, "learning_rate": 0.01, "momentum": 0.9,
        #      "scheduler": {"type": "exponential", "gamma": 0.95},
        #  },
        #  "experiment_name": "DistilCNN_RN56",
        #  },

        # {"dataset": "CIFAR10",
        #  "teacher": {
        #      "model": "WideResNet", "dataset": "CIFAR10", "num_filters": NUM_FILTERS, "num_blocks": 4,
        #      "skip": True, "widen_factor": 4, "ckpt": "lab1/ckpts/WideResNet/WideResNet26-4.pt", "device": "cuda",
        #  },
        #  "student": {
        #      "model": "CNN", "dataset": "CIFAR10", "num_filters": 32, "num_blocks": 1,
        #      "skip": True, "device": "cuda", "weight_decay": 5e-4, "learning_rate": 0.01, "momentum": 0.9,
        #      "scheduler": {"type": "exponential", "gamma": 0.95},
        #  },
        #  "experiment_name": "DistilCNN_WRN26-4",
        #  }

    ]

    for train_params_dict in train_new_configs:
        gen_configs_train(train_params_dict)
        print()
    for distil_params_dict in distil_new_configs:
        gen_configs_distil(distil_params_dict)
        print()
    print("Done")
