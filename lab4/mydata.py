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

    def __init__(self, train=True, ood_set=OOD1):
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
        ood_labels = [dataset.class_to_idx[k] for k in ood_set]
        indices = [i for i, label in enumerate(
            dataset.targets) if label in ood_labels]
        self.aquatic_mammals = Subset(dataset, indices)
        # TODO: map labels in 0-9

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

        if opts.ood_set == "aquatic":
            # CIFAR100 aquatic mammals subset
            ood_set = MyCIFAR100(train=True, ood_set=OOD1)  # 2500 samples
        elif opts.ood_set == "people":
            # CIFAR100 people subset
            ood_set = MyCIFAR100(train=True, ood_set=OOD2)
        elif opts.ood_set == "noise":
            # FakeData (gaussian data)
            # ToTensor() automatically puts data in [0,1]
            transform = transforms.Compose([transforms.ToTensor()])
            ood_set = datasets.FakeData(2500, (3, 28, 28), transform=transform)
        else:
            raise ValueError(f"Unknown ood set {opts.ood_set}")

        ood_loader = make_loader(opts, ood_set)

        return id_loader, ood_loader


def main(opts):
    """Inspect ID and OOD data"""
    # Load data
    train_loader = get_loaders(opts, train=True)
    id_loader, ood_loader = get_loaders(opts, train=False)
    os.makedirs("lab4/plots", exist_ok=True)

    # Inspect train data for AutoEncoder
    train_imgs, train_lab = next(iter(train_loader))
    print("Train data for AutoEncoder (CIFAR10 trainset)")
    print(train_imgs.shape, train_lab.shape)
    print(f"Images: min={train_imgs.min()}, max={train_imgs.max()}")
    print(f"Labels uniques={train_lab.unique()}")
    print()

    output_dir = "lab4/plots"
    # Inspect ID data
    print("ID data (CIFAR10 testset)")
    id_imgs, id_lab = next(iter(id_loader))
    print(id_imgs.shape, id_lab.shape)
    grid = make_grid(id_imgs, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    output_path = os.path.join(output_dir, "id_imgs.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Printed img={output_path}")
    print(f"Images: min={id_imgs.min()}, max={id_imgs.max()}")
    print(f"Labels uniques={id_lab.unique()}")
    print()

    # Inspect OOD data
    print(f"OOD data ({opts.ood_set})")
    ood_imgs, ood_lab = next(iter(ood_loader))
    print(ood_imgs.shape, ood_lab.shape)
    grid = make_grid(ood_imgs, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    output_path = os.path.join(output_dir, opts.ood_set, "ood_imgs.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Printed img={output_path}")
    print(f"Images: min={ood_imgs.min()}, max={ood_imgs.max()}")
    print(f"Labels uniques={ood_lab.unique()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood_set", "--ood", type=str, default="noise",
                        choices=["aquatic", "people", "noise"],
                        help="Choose with OOD dataset to use")
    args = parser.parse_args()
    args.seed = 42
    args.batch_size = 64
    args.num_workers = 2
    try:
        main(args)
    except Exception as e:
        print(e)
