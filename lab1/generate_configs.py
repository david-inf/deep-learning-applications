""" Script for generating YAML configuration files for experiments """

import os
import yaml
from ipdb import launch_ipdb_on_exception


MODELS = ("BaseMLP", "SkipMLP", "BaseCNN", "SkipCNN")

def generate_config(new_params):
    # new_params may contain updated and new parameters
    # Load base configuration file
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)  # dict

    # Automate few parameters
    new_params["experiment_name"] = f"{new_params["model_name"]}"

    # Update with new parameters
    base_config.update(new_params)

    # Dump new configuration file
    output_dir = "experiments"
    os.makedirs(output_dir, exist_ok=True)

    fname = new_params["experiment_name"] + ".yaml"
    output_path = os.path.join(output_dir, fname)
    with open(output_path, "w") as f:
        yaml.dump(base_config, f)

    print(f"Generated config: {output_path}")
    print(f"Parameters: {new_params}")
    print(f"Experiment name: {new_params["experiment_name"]}")


if __name__ == "__main__":
    num_workers = 2
    # add layer sizes
    configs = {
        "BaseMLP": {
            "model_name": "BaseMLP",
            "num_epochs": 20,
            "learning_rate": 0.01,
            "num_workers": num_workers,
        },
        "SkipMLP": {
            "model_name": "SkipMLP",
            "num_epochs": 20,
            "learning_rate": 0.01,
            "num_workers": num_workers,
        },
        "BaseCNN": {
            "model_name": "BaseCNN",
            "num_epochs": 20,
            "learning_rate": 0.01,
            "num_workers": num_workers,
        },
        "SkipCNN": {
            "model_name": "SkipCNN",
            "num_epochs": 20,
            "learning_rate": 0.01,
            "num_workers": num_workers,
        },
    }
    for model, params in configs.items():
        with launch_ipdb_on_exception():
            print(f"Generating configurations for {model}")
            generate_config(params)
            print()
    print("Done")
