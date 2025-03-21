
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ Building block for MLPs """

    def __init__(self, in_features=512, out_features=512, skip=False):
        super().__init__()
        self.skip = skip
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        if self.skip:
            out += identity
        out = self.relu(out)

        return out


class MLP(nn.Module):
    """ Simple MLP with variable layers and optional skip connections """

    def __init__(self, input_size, hidden_size=512, n_blocks=2, skip=False, num_classes=10):
        super().__init__()

        self.input_adapter = nn.Linear(input_size, hidden_size)
        self.mlp = nn.Sequential(
            *[BasicBlock(hidden_size, hidden_size, skip) for _ in range(n_blocks)]
        )
        self.flatten = nn.Flatten()
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # N x C x 28 x 28
        x = self.flatten(x)
        x = self.input_adapter(x)  # hidden_size
        x = self.mlp(x)  # blocks
        x = self.head(x)  # logits
        return x



def build_mlp(opts):
    if opts.dataset.lower() == "mnist":
        input_data = torch.randn(128, 1, 28, 28)
        input_size = 28*28*1
    elif opts.dataset.lower() == "cifar10":
        input_data = torch.randn(128, 3, 28, 28)
        input_size = 28*28*3

    n_blocks = opts.n_blocks if hasattr(opts, "n_blocks") else 2
    hidden_size = opts.hidden_size if hasattr(opts, "hidden_size") else 512
    skip = opts.skip if hasattr(opts, "skip") else False
    model = MLP(input_size, n_blocks=n_blocks,
                hidden_size=hidden_size, skip=skip)

    return model, input_data


def main(opts):
    model, input_data = build_mlp(opts)
    visualize(model, f"{opts.model_name} on {opts.dataset}", input_data)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    import argparse
    import yaml
    from types import SimpleNamespace

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)  # dict
    opts = SimpleNamespace(**configs)

    with launch_ipdb_on_exception():
        main(opts)
