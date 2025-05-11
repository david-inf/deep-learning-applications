"""MLP models"""

import torch
from torch import nn


class MLP(nn.Module):
    """Simple MLP with variable layers"""

    def __init__(self, input_size, layer_sizes=None, num_classes=10):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [128]

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

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)

        x = self.input_adapter(x)  # hidden_size
        x = self.mlp(x)  # blocks

        x = self.classifier(x)  # logits
        return x
