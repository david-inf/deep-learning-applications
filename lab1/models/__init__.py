
import torch
from models.mlp import MLP
from models.cnn import CNN
from models.resnet import ResNet
from models.wideresnet import WideResNet
from utils.misc_utils import LOG


def build_mlp(opts):
    """Whichever MLP you want"""
    if opts.dataset.lower() == "mnist":
        input_data = torch.randn(128, 1, 28, 28)
        input_size = 28*28*1
    elif opts.dataset.lower() == "cifar10":
        input_data = torch.randn(128, 3, 28, 28)
        input_size = 28*28*3
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")

    layers = opts.layers if hasattr(opts, "layers") else [128]
    model = MLP(input_size, layer_sizes=layers)

    return model, input_data


def build_cnn(opts):
    """
    - 2-layer CNN
    - 3-layer ResNet
    - 3-layer WideResNet
    """
    if opts.dataset.lower() == "mnist":
        in_channels = 1
        input_data = torch.randn(128, 1, 28, 28)
    elif opts.dataset.lower() == "cifar10":
        in_channels = 3
        input_data = torch.randn(128, 3, 28, 28)
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")

    num_blocks = opts.num_blocks if hasattr(opts, "num_blocks") else 1
    num_filters = opts.num_filters if hasattr(opts, "num_filters") else 16
    skip = opts.skip if hasattr(opts, "skip") else False
    LOG.info("num_blocks=%s, num_filters=%s, skip=%s",
             num_blocks, num_filters, skip)

    if opts.model == "CNN":
        model = CNN(in_channels, num_filters, num_blocks, skip)
    elif opts.model in ("ResNet", "RN"):
        model = ResNet(in_channels, num_filters, num_blocks, skip)
    elif opts.model in ("WideResNet", "WRN"):
        # widen_factor = opts.widen_factor if hasattr(opts, "widen_factor") else 1
        model = WideResNet(in_channels, num_filters, num_blocks, opts.widen_factor)
    else:
        raise ValueError(f"Unknown model {opts.model_name}")

    return model, input_data
