
import torch
from torch import nn
# from torchvision.models import resnet18


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """ 3x3 convolution """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    """ 1x1 convolution """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class Shortcut(nn.Module):
    """Shortcut ensures dimension matching over the residual block"""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = conv1x1(in_channels, out_channels, stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    """Building block consisting of two 3x3 convolutions with batch norm and relu"""

    def __init__(self, in_channels, out_channels, stride=1, skip=False):
        super().__init__()
        self.skip = skip  # whether to add a skip connection or not

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        if self.skip:
            out += self.shortcut(x)
        out = self.relu(out)

        return out


class CNN(nn.Module):
    """
    2-layers version of ResNet (less depth)
    Total layers: (2*2)*num_blocks + 2
    - First layer as input adapter
    - Sequence of 2 layers: num_blocks * BasicBlock (2 conv layers)
    - Classifier linear layer
    """

    def __init__(self, in_channels=3, num_filters=16, num_blocks=1, skip=False, num_classes=10):
        super().__init__()
        self.in_filters = num_filters
        self.skip = skip

        self.input_adapter = nn.Sequential(
            conv3x3(in_channels, num_filters, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(num_filters*1, num_blocks, stride=2)
        self.layer2 = self._make_layer(num_filters*2, num_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(num_filters*2, num_classes)

    def _make_layer(self, out_filters, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(
                self.in_filters, out_filters, s, self.skip))
            self.in_filters = out_filters
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.input_adapter(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
