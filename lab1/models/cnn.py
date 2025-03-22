
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """ Shortcut ensures dimension matching over the residual block """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = conv1x1(in_channels, out_channels, stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, skip=False):
        super().__init__()
        self.skip = skip  # wether to add skip connection or not

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        if self.skip:
            out += self.shortcut(x)
        out = self.relu(out)

        return out


class CNN(nn.Module):
    def __init__(self, in_channels, n_blocks=1, num_filters=16, num_classes=10, skip=False):
        super().__init__()
        # Adapt channels
        self.input_adapter = nn.Sequential(
            conv3x3(in_channels, num_filters, stride=2, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        blocks = []  # append BasicBlocks objects
        channels = [num_filters]
        # ch -> ch*2 -> ch*4 -> ch*8
        channels += [num_filters * 2**(i+1) for i in range(n_blocks)]
        # list channels has size 1+n_blocks
        for i in range(n_blocks):
            # stride=2 for downsampling layers and 1 otherwise
            stride = 2 if i == 0 else 1
            # TODO: add another intermediate downsampling?
            # stride = 2 if i % 3 == 0 else 1
            blocks.append(BasicBlock(channels[i], channels[i+1],
                                     stride=stride, skip=skip))
        self.blocks = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # N x C x 28 x 28
        x = self.input_adapter(x)  # ; print(x.shape)
        # N x num_filters x 28 x 28
        x = self.blocks(x)  # ; print(x.shape)
        # N x ... x H x W
        x = self.avgpool(x)  # ; print(x.shape)
        # N x ... x 1 x 1
        x = self.flatten(x)  # ; print(x.shape)
        # N x ...
        x = self.fc(x)
        # N x K logits
        return x


def build_cnn(opts):
    if opts.dataset.lower() == "mnist":
        in_channels = 1
        input_data = torch.randn(128, 1, 28, 28)
    elif opts.dataset.lower() == "cifar10":
        in_channels = 3
        input_data = torch.randn(128, 3, 28, 28)

    n_blocks = opts.n_blocks if hasattr(opts, "n_blocks") else 1
    num_filters = opts.num_filters if hasattr(opts, "num_filters") else 32
    skip = opts.skip if hasattr(opts, "skip") else False
    model = CNN(in_channels, n_blocks=n_blocks,
                num_filters=num_filters, skip=skip)

    return model, input_data


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="CNN model")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="Dataset to use for training")
    parser.add_argument("--n_blocks", type=int, default=1,
                        help="Number of residual blocks")
    parser.add_argument("--num_filters", type=int, default=8,
                        help="Number of filters in each block")
    parser.add_argument("--skip", action="store_true",
                        help="Add skip connections")
    return parser.parse_args()


if __name__ == "__main__":
    from torchinfo import summary
    opts = parse_args()
    model, input_data = build_cnn(opts)
    # Count parameters
    model_stats = summary(model, verbose=0)
    print(f"Params: {model_stats.total_params/1e6:.2f}M")
