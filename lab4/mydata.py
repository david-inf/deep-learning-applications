"""CIFAR10 and CIFAR100 wrappers"""

import os
import random

import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, Subset, DataLoader

import matplotlib.pyplot as plt

from utils import set_seeds


ID_CLASSES = (  # CIFAR10
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
OOD_CLASSES = (  # CIFAR100 aquatic mammals
    "beaver", "dolphin", "otter", "seal", "whale")


class MyCIFAR10(Dataset):
    """Wrapper for CIFAR10 Dataset class"""

    def __init__(self, train=True):
        self.num_classes = 10

        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.489255, 0.475775, 0.439889),
                                 std=(0.243047, 0.239315, 0.255997)),
        ])
        self.dataset = datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        returns:
            X: [C, W, H] (tensor)
            y: [] (tensor)
        """
        image, target = self.dataset[idx]
        return image, target


class MyCIFAR100(Dataset):
    """
    Wrapper for CIFAR100 Dataset class
    - aquatic mammals superclass
    """

    def __init__(self, train=True):
        self.num_classes = 5

        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.507075, 0.486548, 0.440917),
                                 std=(0.267334, 0.256438, 0.276150)),
        ])
        dataset = datasets.CIFAR100(
            root="./data", train=train, download=True, transform=transform)

        # Get the aquatic mammals subset
        ood_labels = [dataset.class_to_idx[k] for k in OOD_CLASSES]
        indices = [i for i, label in enumerate(
            dataset.targets) if label in ood_labels]
        self.aquatic_mammals = Subset(dataset, indices)

    def __len__(self):
        return len(self.aquatic_mammals)

    def __getitem__(self, idx):
        image, label = self.aquatic_mammals[idx]
        return image, label


def make_loader(opts, dataset):
    """Build a DataLoader quickly"""
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
    id_testset = MyCIFAR10(train=False)
    id_loader = make_loader(opts, id_testset)

    # OOD testset
    # train: 2500
    # test: 500
    ood_testset = MyCIFAR100(train=False)
    ood_loader = make_loader(opts, ood_testset)

    return id_loader, ood_loader


def main(opts):
    """Inspect ID and OOD data"""
    # Load data
    id_loader, ood_loader = get_loaders(opts)
    os.makedirs("lab4/plots", exist_ok=True)

    # Inspect ID data
    id_imgs, id_lab = next(iter(id_loader))
    print(id_imgs.shape, id_lab.shape)
    grid = make_grid(id_imgs, nrow=4, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    output_path = "lab4/plots/id_imgs.png"
    plt.savefig(output_path)
    print(f"Printed img={output_path}")

    # Inspect OOD data
    ood_imgs, ood_lab = next(iter(ood_loader))
    print(ood_imgs.shape, ood_lab.shape)
    grid = make_grid(ood_imgs, nrow=4, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    output_path = "lab4/plots/ood_imgs.png"
    plt.savefig(output_path)
    print(f"Printed img={output_path}")


if __name__ == "__main__":
    from types import SimpleNamespace
    configs = {
        "seed": 42, "batch_size": 32, "num_workers": 2
    }
    opts = SimpleNamespace(**configs)
    main(opts)
