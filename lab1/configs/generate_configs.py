""" Script for generating YAML configuration files for experiments """

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

# TODO: distillation configs
# def gen_configs_distil(new_params):
#     """Generate a configuration file given the params dict"""
    # _dataset = base_config["dataset"].lower()
    #     if base_config["model_name"] == "Distill":
    #         output_dir = "experiments/distill"
    #         os.makedirs(output_dir, exist_ok=True)
    #         _teacher = base_config["teacher"]
    #         _teacher_prefix = _teacher["model_name"] + count_params(_teacher)
    #         _student = base_config["student"]
    #         _student_prefix = _student["model_name"] + count_params(_student)
    #         _prefix = "Distill_" + _teacher_prefix + \
    #             "_" + _student_prefix

def gen_configs_train(new_params):
    """Generate a configuration file given the params dict"""
    # new_params may contain updated and new params
    # Load base configuration file
    configs = {
        "seed": 42, "dataset": "MNIST", "batch_size": 128, "num_workers": 4,
        "num_epochs": 10, "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 5e-4,
        "scheduler": {"type": "multi-step", "steps": [50, 100], "gamma": 0.1},
        "log_every": 20, "checkpoint": None,
        "do_early_stopping": False,
        "comet_project": "deep-learning-applications", "experiment_key": None,
    }
    configs.update(new_params)

    # Experiment naming
    if configs.get("experiment_name") is None:
        _prefix = configs["model"]# + count_params(configs)
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

    # Dump new configuration file
    fname = configs["experiment_name"] + ".yaml"
    output_dir = os.path.join("lab1/configs", configs["model"])
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(configs, f)

    print(f"Generated file: {output_path}")
    print(f"Configs: {configs}")
    print(f"Experiment: {configs["experiment_name"]}")


if __name__ == "__main__":
    new_configs = [  # list of params (dict)
        # # MLP Tiny
        {"model": "MLP", "dataset": "MNIST",
         "layers": [128, 128], "weight_decay": 0.},
        # # MLP Large
        # {"model_name": "MLP", "dataset": "MNIST", "augmentation": True,
        #  "layers": [1024,1024,512,128], "early_stopping": {"patience": 3, "threshold": 0.01}},
        # {"model_name": "MLP", "dataset": "CIFAR10", "layers": [512], "weight_decay": 0.}

        ## **** ##

        # {"dataset": "MNIST", "model_name": "Distill",
        #  "teacher": {
        #      "model_name": "MLP",
        #      "dataset": "MNIST",
        #      "layers": [1024,1024,512,128],
        #      "ckp": "checkpoints/MLP/e_007_MLP2.45M_mnist_best.pt",
        #      "device": "cuda",
        #  },
        #  "student": {
        #      "model_name": "MLP",
        #      "dataset": "MNIST",
        #      "layers": [64],
        #      "device": "cuda",
        #  },
        #  "weight_decay": 0.,
        #  "learning_rate": 0.01
        # }

        # # CNN Tiny - 0.02M - For distillation
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False, "augmentation": True,
        # "num_epochs": 10, "weight_decay": 0.0001, "learning_rate": 0.1, "lr_decay": 0.9,
        # "num_workers": 4, "num_filters": 16, "num_blocks": 1, "size_type": "tiny"},
        # # CNN Small - 0.07M
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False, "augmentation": True,
        # "num_epochs": 10, "weight_decay": 0.0001, "learning_rate": 0.1, "lr_decay": 0.9,
        # "num_workers": 4, "num_filters": 16, "num_blocks": 3, "size_type": "small"},
        # # CNN Medium - 0.11M
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False, "augmentation": True,
        # "num_epochs": 10, "weight_decay": 0.0001, "learning_rate": 0.1, "lr_decay": 0.9,
        # "num_workers": 4, "num_filters": 16, "num_blocks": 5, "size_type": "medium"},
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": True, "augmentation": True,
        # "num_epochs": 10, "weight_decay": 0.0001, "learning_rate": 0.1, "lr_decay": 0.9,
        # "num_workers": 4, "num_filters": 16, "num_blocks": 5, "size_type": "medium"},
        # # CNN Large - 0.16M
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False, "augmentation": True,
        # "num_epochs": 10, "weight_decay": 0.0001, "learning_rate": 0.1, "lr_decay": 0.9,
        # "num_workers": 4, "num_filters": 16, "num_blocks": 7, "size_type": "large"},
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": True, "augmentation": True,
        #  "Num_epochs": 10, "weight_decay": 0.0001, "learning_rate": 0.1, "lr_decay": 0.9,
        #  "num_workers": 4, "num_filters": 16, "num_blocks": 7, "size_type": "large"},
        # {"model_name": "ResNet", "dataset": "CIFAR10", "skip": True, "augmentation": True,
        #  "num_epochs": 20, "weight_decay": 0.0001, "learning_rate": 0.1, "lr_decay": 0.9,
        #  "num_workers": 4, "num_filters": 32, "num_blocks": 5, "size_type": "large",
        #  "early_stopping": {"patience": 2, "threshold": 0.01}},

        ## **** ##

        # {"dataset": "CIFAR10", "model_name": "Distill",
        #  "teacher": {
        #      "model_name": "ResNet",
        #      "dataset": "CIFAR10",
        #      "num_filters": 32,
        #      "num_blocks": 5,
        #      "skip": True,
        #      "ckp": "checkpoints/ResNet/e_032_ResNet1.86M_cifar10_best.pt",
        #      "device": "cuda",
        #  },
        #  "student": {
        #      "model_name": "CNN",
        #      "dataset": "CIFAR10",
        #      "num_filters": 16,
        #      "num_blocks": 3,
        #      "skip": False,
        #      "device": "cuda",
        #  },
        #   "weight_decay": 0., "learning_rate": 0.1, "lr_decay": 0.9,
        #  }
    ]

    for params_dict in new_configs:
        gen_configs_train(params_dict)
        print()
    print("Done")
