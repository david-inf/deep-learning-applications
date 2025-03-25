
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


# class CNN(nn.Module):
#     def __init__(self, in_channels, n_blocks=1, num_filters=16, num_classes=10, skip=False):
#         super().__init__()
#         self.num_filters = num_filters

#         self.input_adapter = nn.Sequential(
#             conv3x3(in_channels, num_filters, stride=2, padding=1),
#             nn.BatchNorm2d(num_filters),
#             nn.ReLU(inplace=True),
#         )

#         blocks = []  # append BasicBlocks objects
#         channels = [num_filters]
#         # ch -> ch*2 -> ch*4 -> ch*8
#         channels += [num_filters * 2**(i+1) for i in range(n_blocks)]
#         # list channels has size 1+n_blocks
#         for i in range(n_blocks):
#             # stride=2 for downsampling layers and 1 otherwise
#             stride = 2 if i == 0 else 1
#             # TODO: add another intermediate downsampling?
#             # stride = 2 if i % 3 == 0 else 1
#             # TODO: in ResNet we just stack more of this layers
#             # with fixed num_filters
#             blocks.append(BasicBlock(channels[i], channels[i+1],
#                                      stride=stride, skip=skip))
#         self.blocks = nn.Sequential(*blocks)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(channels[-1], num_classes)

#     def forward(self, x):
#         # N x C x 28 x 28
#         x = self.input_adapter(x)  # ; print(x.shape)
#         # N x num_filters x 28 x 28
#         x = self.blocks(x)  # ; print(x.shape)
#         # N x ... x H x W
#         x = self.avgpool(x)  # ; print(x.shape)
#         # N x ... x 1 x 1
#         x = self.flatten(x)  # ; print(x.shape)
#         # N x ...
#         x = self.fc(x)
#         # N x K logits
#         return x


class CNN(nn.Module):
    """
    Total layers: 6*num_blocks + 2
    - First layer as input adapter
    - Sequence of 3 layers: num_blocks * BasicBlock (2 conv layers)
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
        # self.layer3 = self._make_layer(num_filters*4, num_blocks, stride=2)

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
        # x = self.layer3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):
    """ The actual ResNet for CIFAR10 """

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
        self.layer3 = self._make_layer(num_filters*4, num_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(num_filters*4, num_classes)

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
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="CNN model")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="Dataset to use for training")
    parser.add_argument("--num_blocks", type=int, default=1,
                        help="Number of residual blocks")
    parser.add_argument("--num_filters", type=int, default=16,
                        help="Number of filters in each block")
    parser.add_argument("--skip", action="store_true",
                        help="Add skip connections")
    return parser.parse_args()


def compute_flops(model, input_data, epochs=100, batches=390):
    from torchinfo import summary
    model_stats = summary(model, input_data=input_data, verbose=0)

    # Parameters
    print(f"Params: {model_stats.total_params/1e6:.2f}M")

    # forward pass over each batch
    forward_flops = 2 * model_stats.total_mult_adds  # floating-point operations
    # forward + backward pass over each batch -> FLOP/batch
    flops_batch = 3 * forward_flops
    print(f"Training cost per batch: {flops_batch/1e12:.4f} TFLOP")
    flops_epoch = flops_batch * batches  # batches per epoch
    print(f"Training cost per epoch: {flops_epoch/1e12:.4f} TFLOP")
    total_flops = epochs * flops_epoch
    print(f"Training cost: {total_flops/1e12:.4f} TFLOP")

    gpu_flops = 7 * 1e12  # TFLOPS (per second)
    efficiency = .2
    print(f"Training time: {total_flops/gpu_flops/efficiency:.2f} seconds")


if __name__ == "__main__":
    opts = parse_args()
    model, input_data = build_cnn(opts)
    print(f"Total layers: {4*opts.num_blocks+2}")
    compute_flops(model, input_data)
