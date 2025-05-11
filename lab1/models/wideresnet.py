
import torch
import torch.nn as nn
from models.cnn import conv1x1, conv3x3


class Shortcut(nn.Module):
    """Shortcut ensures dimension matching over the residual block"""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv1x1(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor):
        return self.conv(self.relu(self.bn(x)))


class BasicBlock(nn.Module):
    """Wide-dropout block"""

    def __init__(self, in_channels, out_channels, stride=1, droprate=0.):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=droprate, inplace=True)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(self.relu(self.bn2(out)))

        return torch.add(self.shortcut(x), out)


class WideResNet(nn.Module):
    def __init__(self, in_channels=3, num_filters=16, num_blocks=1, widen_factor=1, droprate=0., num_classes=10):
        super().__init__()
        self.in_filters = num_filters
        self.droprate = droprate

        self.conv1 = conv3x3(in_channels, num_filters, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters*4*widen_factor)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            num_filters*1*widen_factor, num_blocks, stride=2)
        self.layer2 = self._make_layer(
            num_filters*2*widen_factor, num_blocks, stride=2)
        self.layer3 = self._make_layer(
            num_filters*4*widen_factor, num_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(num_filters*4*widen_factor, num_classes)

    def _make_layer(self, out_filters, num_blocks, stride):
        """Creating blocks for the current layer"""
        # different stride only for the first BasicBlock
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []  # sequence of BasicBlock

        for stride in strides:
            # Create block and append to blocks list
            blocks.append(
                BasicBlock(self.in_filters, out_filters, stride, self.droprate))
            # Update in_channels for next layer
            self.in_filters = out_filters

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # only conv3x3

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.bn1(x))  # rest of the input_adapter

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)  # logits

        return x
