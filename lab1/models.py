
import torch
import torch.nn as nn


## *********************************** ##

class MLP(nn.Module):
    """ Simple MLP for MNIST with variable layers """
    def __init__(self, layer_sizes=[128], num_classes=10):
        super(MLP, self).__init__()

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
        # N x 784
        x = self.mlp(x)
        x = self.head(x)  # logits
        return x

## *********************************** ##

def main(opts):
    from utils import visualize
    if opts.model_name == "MLP":
        model = MLP([128])

    input_data = torch.randn(64, 1, 28, 28)
    visualize(model, opts.model_name, input_data)

if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    from types import SimpleNamespace

    config = dict(model_name="MLP")
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
