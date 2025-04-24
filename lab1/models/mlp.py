
import torch
import torch.nn as nn


class MLP(nn.Module):
    """ Simple MLP with variable layers """

    def __init__(self, input_size, layer_sizes=[128], num_classes=10):
        super().__init__()

        self.flatten = nn.Flatten()

        self.input_adapter = nn.Sequential(
            nn.Linear(input_size, layer_sizes[0]),
            nn.ReLU(inplace=True)
        )

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

        self.classifier = nn.Linear(layer_sizes[-1], num_classes)

    def forward(self, x):
        x = self.flatten(x)

        x = self.input_adapter(x)  # hidden_size
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
