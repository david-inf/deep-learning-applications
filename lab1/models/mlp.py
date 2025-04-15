
import torch
import torch.nn as nn


class MLP(nn.Module):
    """ Simple MLP with variable layers """

    def __init__(self, input_size, layer_sizes=[128], num_classes=10):
        super().__init__()

        self.flatten = nn.Flatten()

        self.input_adapter = nn.Linear(input_size, layer_sizes[0])
        self.relu = nn.ReLU(inplace=True)

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

        self.classifier = nn.Linear(layer_sizes[-1], num_classes)

    def forward(self, x):
        x = self.flatten(x)

        x = self.relu(self.input_adapter(x))  # hidden_size
        x = self.mlp(x)  # blocks

        x = self.classifier(x)  # logits
        return x


def build_mlp(opts):
    if opts.dataset.lower() == "mnist":
        input_data = torch.randn(128, 1, 28, 28)
        input_size = 28*28*1
    elif opts.dataset.lower() == "cifar10":
        input_data = torch.randn(128, 3, 28, 28)
        input_size = 28*28*3

    layers = opts.layers if hasattr(opts, "layers") else [128]
    model = MLP(input_size, layer_sizes=layers)

    return model, input_data


def parse_args():
    import argparse

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description="MLP model")
    parser.add_argument("--dataset", type=str, default="MNIST",
                        help="Dataset to use for training")
    parser.add_argument("--layers", type=list_of_ints,
                        help="Hidden units for each layer")
    return parser.parse_args()


if __name__ == "__main__":
    from torchinfo import summary
    from ipdb import launch_ipdb_on_exception
    opts = parse_args()
    with launch_ipdb_on_exception():
        model, input_data = build_mlp(opts)
    # Count parameters
    model_stats = summary(model, verbose=0)
    print(f"Params: {model_stats.total_params/1e6:.2f}M")
