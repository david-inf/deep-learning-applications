"""CIFAR10 and CIFAR100 wrappers"""

import sys
import os
import random

import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, Subset, DataLoader

import matplotlib.pyplot as plt

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from lab1.utils import set_seeds
from lab1.mydata import MyCIFAR10


ID_CLASSES = (  # CIFAR10
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

OOD1 = ("beaver", "dolphin", "otter", "seal", "whale")  # aquatic mammals
OOD2 = ("baby", "boy", "girl", "man", "woman")  # people


class MyCIFAR100(Dataset):
    """CIFAR100: aquatic mammals subset"""

    def __init__(self, train=True):
        self.num_classes = 5

        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            # if you want gaussian images
            # transforms.Normalize(mean=(0.507075, 0.486548, 0.440917),
            #                      std=(0.267334, 0.256438, 0.276150)),
        ])
        dataset = datasets.CIFAR100(
            root="./data", train=train, download=True, transform=transform)

        # Get the aquatic mammals subset
        ood_labels = [dataset.class_to_idx[k] for k in OOD2]
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


def get_loaders(opts, train=False):
    """Get data loaders for ID and OOD data"""
    if train:
        # return the trainset for the AutoEncoder
        trainset = MyCIFAR10(opts, train=True)
        train_loader = make_loader(opts, trainset)

        return train_loader
    else:
        # ID testset
        id_set = MyCIFAR10(opts, train=False)
        id_loader = make_loader(opts, id_set)

        # 1) CIFAR100 aquatic mammals subset or other subsets
        # ood_set = MyCIFAR100(train=True)  # 2500 samples
        # 2) FakeData (gaussian data)
        transform = transforms.Compose([transforms.ToTensor()])
        ood_set = datasets.FakeData(2500, (3, 28, 28), transform=transform)

        ood_loader = make_loader(opts, ood_set)

        return id_loader, ood_loader


def main(opts):
    """Inspect ID and OOD data"""
    # Load data
    train_loader = get_loaders(opts, train=True)
    id_loader, ood_loader = get_loaders(opts)
    os.makedirs("lab4/plots", exist_ok=True)

    # Inspect train data for AutoEncoder
    train_imgs, train_lab = next(iter(train_loader))
    print("Train data for AutoEncoder (CIFAR10 trainset)")
    print(train_imgs.shape, train_lab.shape)
    print(f"Min: {train_imgs.min()}, Max: {train_imgs.max()}")
    print()

    # Inspect ID data
    print("ID data (CIFAR10 testset)")
    id_imgs, id_lab = next(iter(id_loader))
    print(id_imgs.shape, id_lab.shape)
    grid = make_grid(id_imgs, nrow=4, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    output_path = "lab4/plots/id_imgs.png"
    plt.savefig(output_path)
    print(f"Printed img={output_path}")
    print(f"Min: {id_imgs.min()}, Max: {id_imgs.max()}")
    print()

    # Inspect OOD data
    print("OOD data")
    ood_imgs, ood_lab = next(iter(ood_loader))
    print(ood_imgs.shape, ood_lab.shape)
    grid = make_grid(ood_imgs, nrow=4, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    output_path = "lab4/plots/ood_imgs.png"
    plt.savefig(output_path)
    print(f"Printed img={output_path}")
    print(f"Min: {ood_imgs.min()}, Max: {ood_imgs.max()}")


if __name__ == "__main__":
    from types import SimpleNamespace
    configs = {
        "seed": 42, "batch_size": 32, "num_workers": 2
    }
    opts = SimpleNamespace(**configs)
    main(opts)
