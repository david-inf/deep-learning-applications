"""Script for inspecting models"""

from utils import visualize, get_model, compute_flops


def main(opts):
    print(opts.experiment_name)
    model, input_data = get_model(opts, True)
    input_data = input_data.to(opts.device)

    visualize(model, f"{opts.model_name} on {opts.dataset}", input_data)
    print()
    # total batches considering validation
    batches = 1132 if opts.dataset == "CIFAR10" else 1362
    compute_flops(model, input_data, opts.num_epochs, batches)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    import argparse
    import yaml
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="Inspect a given model")
    parser.add_argument("--config", help="YAML configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
    opts = SimpleNamespace(**configs)

    with launch_ipdb_on_exception():
        main(opts)
