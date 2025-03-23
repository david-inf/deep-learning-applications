""" Script for generating YAML configuration files for experiments """

import os
from torchinfo import summary
from utils import get_model


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
        with open("config-train.yaml", "r") as f:
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
        output_dir = os.path.join("experiments", base_config["model_name"])
        os.makedirs(output_dir, exist_ok=True)
        _prefix = base_config["model_name"] #+ ("skip" if base_config["skip"] else "")
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
        # # MLP Tiny
        # {"model_name": "MLP", "dataset": "MNIST",
        #  "layers": [64], "weight_decay": 0.},
        # # MLP Large
        # {"model_name": "MLP", "dataset": "MNIST",
        #  "layers": [1024,512,128]},

        ## **** ##

        {"dataset": "MNIST", "model_name": "Distill",
         "teacher": {
             "model_name": "MLP",
             "dataset": "MNIST",
             "layers": [1024,512,128],
             "ckp": "checkpoints/MLP/e_020_MLP1.40M_mnist.pt",
             "device": "cuda",
         },
         "student": {
             "model_name": "MLP",
             "dataset": "MNIST",
             "layers": [64],
             "device": "cuda",
         },
         "weight_decay": 0.,
         "learning_rate": 0.001
        }

        # # CNN Tiny - 0.02M - For distillation
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False,
        #  "num_filters": 16, "num_blocks": 1, "size_type": "tiny"},
        # # CNN Small - 0.07M
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False,
        #  "num_filters": 16, "num_blocks": 3, "size_type": "small"},
        # # CNN Medium - 0.11M
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False,
        #  "num_filters": 16, "num_blocks": 5, "size_type": "medium"},
        # # CNN Large - 0.16M
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": False,
        #  "num_filters": 16, "num_blocks": 7, "size_type": "large"},
        # {"model_name": "CNN", "dataset": "CIFAR10", "skip": True,
        #  "num_filters": 8, "n_blocks": 6, "size_type": "huge", "augmentation": True,
        #  "early_stopping": {"patience": 3, "threshold": 0.01}},

        ## **** ##

        # {"dataset": "CIFAR10", "model_name": "Distill",
        #  "teacher": {
        #      "model_name": "CNN",
        #      "dataset": "CIFAR10",
        #      "num_filters": 8,
        #      "n_blocks": 6,
        #      "skip": True,
        #      "ckp": "checkpoints/CNN/e_018_CNNskip4.90M_cifar10_best.pt",
        #      "device": "cuda",
        #  },
        #  "student": {
        #      "model_name": "CNN",
        #      "dataset": "CIFAR10",
        #      "num_filters": 8,
        #      "n_blocks": 1,
        #      "skip": False,
        #      "device": "cuda",
        #  }
        #  }
    ]

    import yaml
    from types import SimpleNamespace
    from ipdb import launch_ipdb_on_exception
    for params in configs:
        with launch_ipdb_on_exception():
            gen_configs(params)
            print()
    print("Done")
