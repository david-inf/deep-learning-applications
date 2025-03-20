
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet18

from models.mlp import visualize


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


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, skip=False):
        super().__init__()
        self.skip = skip  # wether to add skip connection or not

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ensure dimension matching (filters)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.skip:
            identity = self.shortcut(x)
            out += identity
        out = self.relu(out)

        return out


class CNN(nn.Module):
    def __init__(self, in_channels, num_filters=64, num_classes=10, skip=False):
        super().__init__()
        # First convolution
        self.input_adapter = nn.Sequential(
            conv3x3(in_channels, num_filters, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = BasicBlock(
            num_filters, num_filters*2, stride=1, skip=skip)
        self.layer2 = BasicBlock(
            num_filters*2, num_filters*4, stride=2, skip=skip)
        self.layer3 = BasicBlock(
            num_filters*4, num_filters*8, stride=2, skip=skip)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters*8, num_classes)

    def forward(self, x):
        x = self.input_adapter(x)  # ; print(x.shape)
        # N x num_filters x 28 x 28

        x = self.layer1(x)  # ; print(x.shape)
        # N x num_filters*2 x 28 x 28
        x = self.layer2(x)  # ; print(x.shape)
        # N x num_filters*4 x 14 x 14
        x = self.layer3(x)  # ; print(x.shape)
        # N x num_filters*8 x 7 x 7

        x = self.avgpool(x)  # ; print(x.shape)
        # N x num_filters*8 x 1 x 1
        x = self.flatten(x)  # ; print(x.shape)
        # N x num_filters*8
        x = self.fc(x)
        # N x K
        return x


def main(args):
    if args.model == "BaseCNN":
        if args.dataset.lower() == "mnist":
            model = CNN(1, num_filters=64, skip=False)
        elif args.dataset.lower() == "cifar10":
            model = CNN(3, num_filters=64, skip=False)
    elif args.model == "SkipCNN":
        if args.dataset.lower() == "mnist":
            model = CNN(1, num_filters=64, skip=True)
        elif args.dataset.lower() == "cifar10":
            model = CNN(3, num_filters=64, skip=True)

    if args.dataset.lower() == "mnist":
        input_data = torch.randn(128, 1, 28, 28)
    elif args.dataset.lower() == "cifar10":
        input_data = torch.randn(128, 3, 28, 28)

    visualize(model, f"{args.model} on {args.dataset}", input_data)


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
        main(args)
