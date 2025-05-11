
import torch
from torch import nn

from models.cnn import conv3x3, BasicBlock


class ResNet(nn.Module):
    """
    The actual ResNet for CIFAR10
    Total layers: (3*2)*num_blocks + 2
    - First layer as input adapter
    - Sequence of 3 layers: num_blocks * BasicBlock (2 conv layers)
    - Classifier linear layer
    """

    def __init__(self, in_channels=3, num_filters=16, num_blocks=1, skip=True, num_classes=10):
        super().__init__()
        self.in_filters = num_filters  # planes
        self.skip = skip  # add skip connection, compatibility with experiments

        self.input_adapter = nn.Sequential(  # input image downsampling
            conv3x3(in_channels, num_filters, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # sequence of 3 layers with variable number of BasicBlock
        self.layer1 = self._make_layer(num_filters*1, num_blocks, stride=2)
        self.layer2 = self._make_layer(num_filters*2, num_blocks, stride=2)
        self.layer3 = self._make_layer(num_filters*4, num_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(num_filters*4, num_classes)

    def _make_layer(self, out_filters, num_blocks, stride):
        """Creating blocks for the current layer"""
        # different stride only for the first BasicBlock
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []  # sequence of BasicBlock

        for s in strides:
            # Create block and append to blocks list
            blocks.append(
                BasicBlock(self.in_filters, out_filters, s, self.skip))
            # Update in_channels for next layer
            self.in_filters = out_filters

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        x = self.input_adapter(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)  # logits

        return x
