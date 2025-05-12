"""CIFAR10 and CIFAR100 wrappers"""

# TODO: print one image per ID-class from CIFAR10
# TODO: print one image per OOD-class from CIFAR100

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset

from lab1.mydata import MyCIFAR10

CIFAR10_CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
OOD_CLASSES = (
    "beaver", "dolphin", "otter", "seal", "whale")


class MyCIFAR100(Dataset):
    """
    Wrapper for CIFAR100 Dataset class
    - aquatic mammals superclass
    """

    def __init__(self, crop=True, train=True):
        self.num_classes = 5

        dataset = datasets.CIFAR100(
            root="./data", train=train, download=True)
        X, y = dataset.data, dataset.targets

        # TODO: get the aquatic mammals subset

        # Prepare data
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2) / 255.  # [0,1]
        if crop:
            margin = (32 - 28) // 2
            self.X = self.X[:, :, margin:-margin, margin:-margin]
        self.y = torch.tensor(y)

        # Normalize data with given mean and std
        self.mean = torch.tensor([0.5, 0.5, 0.5])
        self.std = torch.tensor([0.5, 0.5, 0.5])
        self.X = transforms.Normalize(self.mean, self.std)(self.X)

    def __len__(self):
        # TODO: compute for the aquatic mammals subset
        return self.X.shape[0]

    # def __getitem__(self, idx):


def get_loaders(opts):
    # ID testset
    id_testset = MyCIFAR10(opts, train=False)
    # OOD testset
    ood_testset = 
