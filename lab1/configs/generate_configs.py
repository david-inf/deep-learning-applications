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
    """Generate a configuration file given a params dict"""
    configs = {
        "seed": 42, "dataset": "CIFAR10", "batch_size": 128, "num_workers": 4,
        "num_epochs": 20, "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 5e-4,
        # "scheduler": {"type": "multi-step", "steps": [50, 100], "gamma": 0.1},
        "scheduler": {"type": "exponential", "gamma": 0.95},
        "log_every": 20, "checkpoint": None,
        "do_early_stopping": False, "augmentation": True,
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
    NUM_FILTERS = 32
    new_configs = [  # list of params (dict)
        # # MLP Tiny
        # {"model": "MLP", "dataset": "MNIST",
        #  "layers": [128, 128], "weight_decay": 0.},
        # # MLP Large
        # {"model_name": "MLP", "dataset": "MNIST", "augmentation": True,
        #  "layers": [1024,1024,512,128], "early_stopping": {"patience": 3, "threshold": 0.01}},
        # {"model_name": "MLP", "dataset": "CIFAR10", "layers": [512], "weight_decay": 0.}

        ## **** ##

        # # SmallCNN - For distillation
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

        # ResNet (teacher model)
        {"model": "ResNet", "num_filters": NUM_FILTERS, "num_blocks": 5, "skip": True,
         "early_stopping": {"patience": 4, "min_delta": 0.002}, "do_early_stopping": True,
         "experiment_name": "ResNet16",
         },
        # WideResNet

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
