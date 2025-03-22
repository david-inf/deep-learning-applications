
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Building block for MLPs

    Args:
        in_features : int
            Input and output number of units
        out_features : int
            Middle number of units
        skip : bool
            Wether to add or not the skip connection
    """

    def __init__(self, in_features, out_features, skip=False):
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
    """
    Simple MLP with variable layers and optional skip connections

    Args:
        input_size : int
            Number of units for the input
        hidden_size : int
            Number of units for each layer of the blocks
        n_blocks : int
            Number of blocks
        skip : bool
            Wether to add or not the skip connections
        num_classes : int
            Number of output classes
    """

    def __init__(self, input_size, hidden_size=64, n_blocks=1, skip=False, num_classes=10):
        super().__init__()

        self.input_adapter = nn.Linear(input_size, hidden_size)
        self.mlp = nn.Sequential(
            # upsampling with the middle hidden unit
            *[BasicBlock(hidden_size, hidden_size*2, skip) for _ in range(n_blocks)]
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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="MLP model")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="Dataset to use for training")
    parser.add_argument("--n_blocks", type=int, default=1,
                        help="Number of residual blocks")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Number of units in each block")
    parser.add_argument("--skip", action="store_true",
                        help="Add skip connections")
    return parser.parse_args()


if __name__ == "__main__":
    from torchinfo import summary
    opts = parse_args()
    model, input_data = build_mlp(opts)
    # Count parameters
    model_stats = summary(model, verbose=0)
    print(f"Params: {model_stats.total_params/1e6:.2f}M")
