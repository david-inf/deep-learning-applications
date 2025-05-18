"""CIFAR10 and CIFAR100 wrappers"""

# TODO: print one image per ID-class from CIFAR10
# TODO: print one image per OOD-class from CIFAR100

import random

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader

from utils import set_seeds


ID_CLASSES = (  # CIFAR10
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
OOD_CLASSES = (  # CIFAR100 aquatic mammals
    "beaver", "dolphin", "otter", "seal", "whale")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

class MyCIFAR10(Dataset):
    """Wrapper for CIFAR10 Dataset class"""

    def __init__(self, crop=True, train=True):
        self.num_classes = 10

        dataset = datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform)
        self.X, self.y = dataset.data, dataset.targets  # ndarray, list

        if crop:
            margin = (32 - 28) // 2
            self.X = self.X[:, :, margin:-margin, margin:-margin]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        image, target = self.X[idx], self.y[idx]
        return image, target


class MyCIFAR100(Dataset):
    """
    Wrapper for CIFAR100 Dataset class
    - aquatic mammals superclass
    """

    def __init__(self, crop=True, train=True):
        self.num_classes = 5

        dataset = datasets.CIFAR100(
            root="./data", train=train, download=True, transform=transform)

        if crop:
            margin = (32 - 28) // 2
            dataset.data = dataset.data[:, :, margin:-margin, margin:-margin]

        # Get the aquatic mammals subset
        ood_labels = [dataset.class_to_idx[k] for k in OOD_CLASSES]
        indices = [i for i, label in enumerate(dataset.targets) if label in ood_labels]
        self.aquatic_mammals = Subset(dataset, indices)

    def __len__(self):
        return len(self.aquatic_mammals)

    def __getitem__(self, idx):
        image, label = self.aquatic_mammals[idx]
        return image, label


def make_loader(opts, dataset):
    set_seeds(opts.seed)
    generator = torch.Generator().manual_seed(opts.seed)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,
        pin_memory=True, generator=generator, worker_init_fn=seed_worker,
    )
    return loader


def get_loaders(opts):
    """Get data loaders for ID and OOD data"""
    # ID testset
    id_testset = MyCIFAR10(opts, train=False)
    id_loader = make_loader(opts, id_testset)

    # OOD testset
    # train: 2500
    # test: 500
    ood_testset = MyCIFAR100(train=False)
    ood_loader = make_loader(opts, ood_testset)

    return id_loader, ood_loader


if __name__ == "__main__":
    from types import SimpleNamespace
    configs = {
        "seed": 42, "batch_size": 128, "num_workers": 2
    }
    opts = SimpleNamespace(**configs)
    get_loaders(opts)
