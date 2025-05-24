"""Data loading utilities"""

import sys
import os
import random

import torch
import numpy as np
from torchvision import datasets
# import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    # Add parent directory to path
    os.path.dirname(os.path.abspath(__file__))))

from lab1.utils.misc import set_seeds


class MyMNIST(Dataset):
    """Wrapper for MNIST Dataset class"""

    def __init__(self, opts, train=True):
        self.opts = opts
        self.num_classes = 10

        dataset = datasets.MNIST(
            root="./data", train=train, download=True)
        images, labels = dataset.data, dataset.targets  # tensor, tensor

        # Prepare data
        self.images = images.unsqueeze(1) / 255.  # [N, 1, 28, 28] data in [0,1]
        self.labels = labels  # [N]

        # Normalize data with given mean and std
        # if you want gaussian data
        # self.mean = torch.tensor([0.130925])
        # self.std = torch.tensor([0.308449])
        # if you want data in [-1,1]
        # self.mean = torch.tensor([0.5])
        # self.std = torch.tensor([0.5])
        # self.X = transforms.Normalize(self.mean, self.std)(self.X)

    def __len__(self):
        return self.images.shape[0]  # 70000

    def __getitem__(self, idx):
        """
        returns:
            X: [C, W, H] (tensor)
            y: [] (tensor)
        """
        image, target = self.images[idx], self.labels[idx]
        return image, target


class MyAugmentedMNIST(MyMNIST):
    """MNIST wrapper with data agumentations"""

    def __init__(self, opts):
        super().__init__(opts, train=True)
        self.augmentation_pipeline = v2.Compose([
            v2.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])

    def __getitem__(self, idx):
        return self.augmentation_pipeline(self.images[idx]), self.labels[idx]


class MyCIFAR10(Dataset):
    """Wrapper for CIFAR10 Dataset class"""

    def __init__(self, opts, crop=True, train=True):
        self.opts = opts
        self.num_classes = 10

        dataset = datasets.CIFAR10(
            root="./data", train=train, download=True)
        images, labels = dataset.data, dataset.targets  # ndarray, list

        # Prepare data
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2) / 255.  # [0,1]
        if crop:
            margin = (32 - 28) // 2
            self.images = self.images[:, :, margin:-margin, margin:-margin]
        self.labels = torch.tensor(labels)

        # Normalize data with given mean and std
        # if you want gaussian data
        # self.mean = torch.tensor([0.489255, 0.475775, 0.439889])
        # self.std = torch.tensor([0.243047, 0.239315, 0.255997])
        # if you want data in [-1,1]
        # self.mean = torch.tensor([0.5, 0.5, 0.5])
        # self.std = torch.tensor([0.5, 0.5, 0.5])
        # self.X = transforms.Normalize(self.mean, self.std)(self.X)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """
        returns:
            X: [C, W, H] (tensor)
            y: [] (tensor)
        """
        image, target = self.images[idx], self.images[idx]
        return image, target


class MyAugmentedCIFAR10(MyCIFAR10):
    """CIFAR10 wrapper with data augmentations"""

    def __init__(self, opts):
        super().__init__(opts, crop=False, train=True)
        self.augmentation_pipeline = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(size=32, padding=4),
        ])

    def __getitem__(self, idx):
        return self.augmentation_pipeline(self.images[idx]), self.labels[idx]


class MakeDataLoaders:
    """
    Create train-val data loaders

    Args:
        opts : SimpleNamespace
        traindata : Dataset
            Could be with or without augmentation
        valdata : Dataset
    """

    def __init__(self, opts, trainset, valset):
        set_seeds(opts.seed)
        generator = torch.Generator().manual_seed(opts.seed)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.train_loader = DataLoader(
            trainset, batch_size=opts.batch_size, shuffle=True,
            num_workers=opts.num_workers, pin_memory=True,
            generator=generator, worker_init_fn=seed_worker
        )
        self.val_loader = DataLoader(
            valset, batch_size=opts.batch_size, num_workers=opts.num_workers,
            pin_memory=True, generator=generator, worker_init_fn=seed_worker
        )


def main_mnist(opts):
    """MNIST dataset inspection"""
    # Full MNIST dataset
    # (X, y) train: N=60000 ; test: N=10000 -> 70000

    # Dataset statistics channel-wise
    print("MNIST original dataset")
    trainset = datasets.MNIST(root="./data", train=True, download=True)
    valset = datasets.MNIST(root="./data", train=False, download=True)
    print(type(trainset.data), trainset.data.shape)
    print(type(valset.data), valset.data.shape)
    print(type(trainset.targets), trainset.targets.shape)
    print(type(valset.targets), valset.targets.shape)
    # merge datasets and compute mean and std
    X = torch.tensor(np.vstack((trainset.data, valset.data))
                     ).unsqueeze(1) / 255.
    y = torch.cat((trainset.targets, valset.targets))
    mean = torch.mean(X, dim=(0, 2, 3))  # original data mean
    std = torch.std(X, dim=(0, 2, 3))  # original data std
    print(f"Dataset size: X={X.shape}, y={y.shape}")
    torch.set_printoptions(precision=6)
    print(f"Mean: {mean}, Std: {std}\nMin: {X.min()}, Max: {X.max()}")
    print()

    # Custom datasets
    trainset_original = MyMNIST(opts)
    trainset_augmented = MyAugmentedMNIST(opts)
    valset = MyMNIST(opts, train=False)

    # Check dataset statistics channel-wise
    X = torch.cat((trainset_original.images, valset.images))
    y = torch.cat((trainset_original.labels, valset.labels))
    mean = torch.mean(X, dim=(0, 2, 3))  # original data mean
    std = torch.std(X, dim=(0, 2, 3))  # original data std
    print("MNIST custom datasets")
    print(f"Dataset size: X={X.shape}, y={y.shape}")
    print(f"Mean: {mean}, Std: {std}\nMin: {X.min()}, Max: {X.max()}")

    # Data loaders
    mnist_original = MakeDataLoaders(opts, trainset_original, valset)
    mnist_augmented = MakeDataLoaders(opts, trainset_augmented, valset)
    train_loader = mnist_original.train_loader
    train_loader_aug = mnist_augmented.train_loader

    # Check first batch
    X_orig, y_orig = next(iter(train_loader))
    X_aug, _ = next(iter(train_loader_aug))
    print("Data shape:", f"X={X_orig.shape}", f"y={y_orig.shape}")

    # Print first batch
    os.makedirs("lab1/plots", exist_ok=True)
    grid = make_grid(X_orig, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("lab1/plots/data/mnist.png")

    grid = make_grid(X_aug, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("lab1/plots/data/mnist_augmented.png")


def main_cifar10(opts):
    # Full CIFAR10 dataset
    # (X, y) train: N=50000 ; test: N=10000 -> 60000
    CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Original dataset
    print("CIFAR10 original dataset")
    trainset = datasets.CIFAR10(root="./data", train=True, download=True)
    valset = datasets.CIFAR10(root="./data", train=False, download=True)
    print(type(trainset.data), trainset.data.shape)
    print(type(valset.data), valset.data.shape)
    print(type(trainset.targets), len(trainset.targets))
    print(type(valset.targets), len(valset.targets))
    # merge datasets and compute meand and std
    X = torch.tensor(np.vstack((trainset.data, valset.data))
                     ).permute(0, 3, 1, 2) / 255.
    margin = (32 - 28) // 2
    X = X[:, :, margin:-margin, margin:-margin]
    y = torch.tensor(trainset.targets + valset.targets)
    mean = torch.mean(X, dim=(0, 2, 3))  # original data mean
    std = torch.std(X, dim=(0, 2, 3))  # original data std
    print(f"Dataset size: X={X.shape}, y={y.shape}")
    torch.set_printoptions(precision=6)
    print(f"Mean: {mean}, Std: {std}\nMin: {X.min()}, Max: {X.max()}")
    print()

    # Custom datasets
    trainset_original = MyCIFAR10(opts)
    trainset_augmented = MyAugmentedCIFAR10(opts)
    valset = MyCIFAR10(opts, train=False)

    # Check dataset statistics channel-wise
    X = torch.cat((trainset_original.images, valset.images))
    y = torch.cat((trainset_original.labels, valset.labels))
    mean = torch.mean(X, dim=(0, 2, 3))  # original data mean
    std = torch.std(X, dim=(0, 2, 3))  # original data std
    print("CIFAR10 custom datasets")
    print(f"Dataset size: X={X.shape}, y={y.shape}")
    print(f"Mean: {mean}, Std: {std}\nMin: {X.min()}, Max: {X.max()}")

    # Data loaders
    cifar10_original = MakeDataLoaders(opts, trainset_original, valset)
    cifar10_augmented = MakeDataLoaders(opts, trainset_augmented, valset)
    train_loader = cifar10_original.train_loader
    train_loader_aug = cifar10_augmented.train_loader

    # Check first batch
    X_orig, y_orig = next(iter(train_loader))
    X_aug, _ = next(iter(train_loader_aug))
    print("Data shape:", f"X={X_orig.shape}", f"y={y_orig.shape}")

    # Print first batch
    os.makedirs("lab1/plots", exist_ok=True)
    grid = make_grid(X_orig, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("lab1/plots/data/cifar10.png")

    grid = make_grid(X_aug, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("lab1/plots/data/cifar10_augmented.png")


if __name__ == "__main__":
    from types import SimpleNamespace

    config = dict(seed=42, val_size=0.1,
                  batch_size=64, num_workers=2)
    opts = SimpleNamespace(**config)
    set_seeds(opts.seed)

    main_mnist(opts)
    print()
    main_cifar10(opts)
