
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
    """Shortcut ensures dimension matching over the residual block"""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = conv1x1(in_channels, out_channels, stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
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

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        if self.skip:
            out += self.shortcut(x)
        out = self.relu(out)

        return out


class InverseBasicBlock(nn.Module):
    """Same as BasicBlock but with pre-activation batch norm and relu"""

    def __init__(self, in_channels, out_channels, stride=1, skip=True):
        super().__init__()
        self.skip = skip

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.relu = nn.ReLU(inplace=True)
        # altrimenti posso fargli ereditare BasicBlock e quindi cambio solo il forward se può aver senso


class CNN(nn.Module):
    """
    2 layers version of ResNet (less deep)
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
        for stride in strides:
            layers.append(BasicBlock(self.in_filters, out_filters,
                                     stride, self.skip))
            self.in_filters = out_filters
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_adapter(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):
    """
    The actual ResNet for CIFAR10
    Total layers: (3*2)*num_blocks + 2
    - First layer as input adapter
    - Sequence of 3 layers: num_blocks * BasicBlock (2 conv layers)
    - Classifier linear layer
    """

    def __init__(self, in_channels=3, num_filters=16, num_blocks=1, skip=True, num_classes=10):
        # TODO: block argument allowing different types of basic blocks
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

        for stride in strides:
            # Create block and append to blocks list
            blocks.append(
                BasicBlock(self.in_filters, out_filters, stride, self.skip))
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


def build_cnn(opts):
    if opts.dataset.lower() == "mnist":
        in_channels = 1
        input_data = torch.randn(128, 1, 28, 28)
    elif opts.dataset.lower() == "cifar10":
        in_channels = 3
        input_data = torch.randn(128, 3, 28, 28)

    num_blocks = opts.num_blocks if hasattr(opts, "num_blocks") else 1
    num_filters = opts.num_filters if hasattr(opts, "num_filters") else 16
    skip = opts.skip if hasattr(opts, "skip") else False

    if opts.model_name == "CNN":
        model = CNN(in_channels, num_filters, num_blocks, skip)
    elif opts.model_name == "ResNet":
        model = ResNet(in_channels, num_filters, num_blocks, skip)
    else:
        raise ValueError(f"Unknown model {opts.model_name}")

    return model, input_data
