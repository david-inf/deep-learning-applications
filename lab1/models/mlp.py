
import torch
import torch.nn as nn
import torch.nn.functional as F


## *********************************** ##

class BaseMLP(nn.Module):
    """ Simple MLP for MNIST with variable layers """
    def __init__(self, layer_sizes=[768]*3, num_classes=10):
        super().__init__()

        sizes = [784] + layer_sizes + [num_classes]
        layers = []
        for size in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[size], sizes[size+1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        # N x 28 x 28
        x = x.flatten(1)
        x = self.mlp(x)
        x = self.head(x)  # logits
        return x

## *********************************** ##

class SkipMLPBlock(nn.Module):
    """ Building block for SkipMLP """
    def __init__(self, in_features=768, out_features=768):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, x):
        f = F.relu(self.fc1(x))
        f = F.relu(self.fc2(f))
        return f + x


class SkipMLP(nn.Module):
    """ MLP with skip connections for residual learning """
    def __init__(self, hidden_size=768, n_blocks=1, num_classes=10):
        super().__init__()

        self.input_adapter = nn.Linear(784, hidden_size)
        self.blocks = nn.Sequential(
            *[SkipMLPBlock(hidden_size, hidden_size)]*n_blocks
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.input_adapter(x))
        x = self.blocks(x)
        x = self.head(x)
        return x

## *********************************** ##

def visualize(model, model_name, input_data):
    from torchinfo import summary
    from rich.console import Console
    out = model(input_data)

    console = Console()
    console.print(f"Model {model_name}, computed output shape = {out.shape}")

    model_stats = summary(
        model,
        input_data=input_data,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            # "params_percent",
            # "kernel_size",
            # "mult_adds",
        ],
        row_settings=("var_names",),
        col_width=18,
        depth=8,
        verbose=0,
    )
    console.print(model_stats)


def main(args):
    if args.model == "BaseMLP":
        model = BaseMLP()
    elif args.model == "SkipMLP":
        model = SkipMLP()

    input_data = torch.randn(64, 1, 28, 28)
    visualize(model, args.model, input_data)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BaseMLP")
    args = parser.parse_args()

    with launch_ipdb_on_exception():
        main(args)
