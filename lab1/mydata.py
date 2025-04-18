
import random
import numpy as np

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset
from utils import set_seeds


class MyMNIST(Dataset):
    """ Wrapper for MNIST Dataset class """

    def __init__(self, opts, train=True):
        # TODO: add data augmentation (maybe another class)
        self.opts = opts
        self.num_classes = 10

        dataset = datasets.MNIST(
            root="../../data", train=train, download=True)
        X, y = dataset.data, dataset.targets  # tensor, tensor

        # Prepare data
        self.X = X.unsqueeze(1) / 255.  # [N, 1, 28, 28] in [0,1]
        self.y = y  # [N]

        # Normalize data with given mean and std
        self.mean = torch.tensor([0.130925])
        self.std = torch.tensor([0.308449])
        self.X = transforms.Normalize(self.mean, self.std)(self.X)

    def __len__(self):
        return self.X.shape[0]  # 70000

    def __getitem__(self, idx):
        """
        returns:
            X: [C, W, H] (tensor)
            y: [] (tensor)
        """
        image, target = self.X[idx], self.y[idx]
        return image, target


class MyAugmentedMNIST(MyMNIST):
    def __init__(self, opts):
        super().__init__(opts, train=True)
        self.augmentation_pipeline = v2.Compose([
            v2.RandomAffine(degrees=15, translate=(
                0.1, 0.1), scale=(0.9, 1.1)),
        ])

    def __getitem__(self, idx):
        return self.augmentation_pipeline(self.X[idx]), self.y[idx]


class MyCIFAR10(Dataset):
    """ Wrapper for CIFAR10 Dataset class """

    def __init__(self, opts, crop=True, train=True):
        self.opts = opts
        self.num_classes = 10

        dataset = datasets.CIFAR10(
            root="../../data", train=train, download=True)
        X, y = dataset.data, dataset.targets  # ndarray, list

        # Prepare data
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2) / 255.  # [0,1]
        if crop:
            margin = (32 - 28) // 2
            self.X = self.X[:, :, margin:-margin, margin:-margin]
        self.y = torch.tensor(y)

        # Normalize data with given mean and std
        self.mean = torch.tensor([0.489255, 0.475775, 0.439889])
        self.std = torch.tensor([0.243047, 0.239315, 0.255997])
        self.X = transforms.Normalize(self.mean, self.std)(self.X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        returns:
            X: [C, W, H] (tensor)
            y: [] (tensor)
        """
        image, target = self.X[idx], self.y[idx]
        return image, target


class MyAugmentedCIFAR10(MyCIFAR10):
    def __init__(self, opts):
        super().__init__(opts, crop=False, train=True)
        self.augmentation_pipeline = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(size=32, padding=4),
        ])

    def __getitem__(self, idx):
        return self.augmentation_pipeline(self.X[idx]), self.y[idx]


class MakeDataLoaders:
    """
    Create train, val, test data loaders

    Args:
        opts : SimpleNamespace
        traindata : Dataset
            Could be with or without augmentation
        valdata : Dataset
        testdata : Dataset
    """

    def __init__(self, opts, traindata, valdata, testdata):
        from torch.utils.data import DataLoader, SubsetRandomSampler
        set_seeds(opts.seed)
        generator = torch.Generator().manual_seed(opts.seed)
        # 1) Dataset objects for train, val and test
        full_trainset = traindata
        fullvalset = valdata
        testset = testdata

        # 2) Train-Val split
        N = len(full_trainset)
        indices = list(range(N))
        split = int(np.floor(opts.val_size * N))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]

        # TODO: consider weighted sampling
        train_sampler = SubsetRandomSampler(train_idx, generator=generator)
        val_sampler = SubsetRandomSampler(val_idx, generator=generator)

        # 3) Data loaders
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.train_loader = DataLoader(
            full_trainset, batch_size=opts.batch_size, sampler=train_sampler,
            shuffle=True, num_workers=opts.num_workers, pin_memory=True,
            generator=generator, worker_init_fn=seed_worker
        )
        self.val_loader = DataLoader(
            fullvalset, batch_size=opts.batch_size, sampler=val_sampler,
            num_workers=opts.num_workers, pin_memory=True,
            generator=generator, worker_init_fn=seed_worker
        )
        self.test_loader = DataLoader(
            testset, batch_size=opts.batch_size,
            num_workers=opts.num_workers, pin_memory=True,
            generator=generator, worker_init_fn=seed_worker
        )


def main_mnist(opts):
    # Full MNIST dataset
    # (X, y) train: N=60000 ; test: N=10000 -> 70000

    # Dataset statistics channel-wise
    print("MNIST original dataset")
    trainset = datasets.MNIST(root="../../data", train=True, download=True)
    testset = datasets.MNIST(root="../../data", train=False, download=True)
    print(type(trainset.data), trainset.data.shape)
    print(type(testset.data), testset.data.shape)
    print(type(trainset.targets), trainset.targets.shape)
    print(type(testset.targets), testset.targets.shape)
    # merge datasets and compute mean and std
    X = torch.tensor(np.vstack((trainset.data, testset.data))
                     ).unsqueeze(1) / 255.
    y = torch.cat((trainset.targets, testset.targets))
    mean = torch.mean(X, dim=(0, 2, 3))  # original data mean
    std = torch.std(X, dim=(0, 2, 3))  # original data std
    print(f"Dataset size: X={X.shape}, y={y.shape}")
    torch.set_printoptions(precision=6)
    print(f"Mean: {mean}, Std: {std}")
    print()

    # Custom datasets
    trainset_original = MyMNIST(opts)
    trainset_augmented = MyAugmentedMNIST(opts)
    valset = MyMNIST(opts)
    testset = MyMNIST(opts, train=False)

    # Check dataset statistics channel-wise
    X = torch.cat((trainset_original.X, testset.X))
    y = torch.cat((trainset_original.y, testset.y))
    mean = torch.mean(X, dim=(0, 2, 3))  # original data mean
    std = torch.std(X, dim=(0, 2, 3))  # original data std
    print("MNIST custom datasets")
    print(f"Dataset size: X={X.shape}, y={y.shape}")
    print(f"Mean: {mean}, Std: {std}")

    # Data loaders
    mnist_original = MakeDataLoaders(opts, trainset_original, valset, testset)
    mnist_augmented = MakeDataLoaders(opts, trainset_augmented, valset, testset)
    train_loader = mnist_original.train_loader
    train_loader_aug = mnist_augmented.train_loader

    # Check first batch
    X_orig, y_orig = next(iter(train_loader))
    X_aug, y_aug = next(iter(train_loader_aug))
    print("Data shape:", f"X={X_orig.shape}", f"y={y_orig.shape}")

    # Print first batch
    import os
    os.makedirs("plots", exist_ok=True)
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    grid = make_grid(X_orig, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("plots/mnist.png")

    grid = make_grid(X_aug, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("plots/mnist_augmented.png")


def main_cifar10(opts):
    # Full CIFAR10 dataset
    # (X, y) train: N=50000 ; test: N=10000 -> 60000
    CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Original dataset
    print("CIFAR10 original dataset")
    trainset = datasets.CIFAR10(root="../../data", train=True, download=True)
    testset = datasets.CIFAR10(root="../../data", train=False, download=True)
    print(type(trainset.data), trainset.data.shape)
    print(type(testset.data), testset.data.shape)
    print(type(trainset.targets), len(trainset.targets))
    print(type(testset.targets), len(testset.targets))
    # merge datasets and compute meand and std
    X = torch.tensor(np.vstack((trainset.data, testset.data))
                     ).permute(0, 3, 1, 2) / 255.
    margin = (32 - 28) // 2
    X = X[:, :, margin:-margin, margin:-margin]
    y = torch.tensor(trainset.targets + testset.targets)
    mean = torch.mean(X, dim=(0, 2, 3))  # original data mean
    std = torch.std(X, dim=(0, 2, 3))  # original data std
    print(f"Dataset size: X={X.shape}, y={y.shape}")
    torch.set_printoptions(precision=6)
    print(f"Mean: {mean}, Std: {std}")
    print()

    # Custom datasets
    trainset_original = MyCIFAR10(opts)
    trainset_augmented = MyAugmentedCIFAR10(opts)
    valset = MyCIFAR10(opts, train=True)
    testset = MyCIFAR10(opts, train=False)

    # Check dataset statistics channel-wise
    X = torch.cat((trainset_original.X, testset.X))
    y = torch.cat((trainset_original.y, testset.y))
    mean = torch.mean(X, dim=(0, 2, 3))  # original data mean
    std = torch.std(X, dim=(0, 2, 3))  # original data std
    print("CIFAR10 custom datasets")
    print(f"Dataset size: X={X.shape}, y={y.shape}")
    print(f"Mean: {mean}, Std: {std}")

    # Data loaders
    cifar10_original = MakeDataLoaders(
        opts, trainset_original, valset, testset)
    cifar10_augmented = MakeDataLoaders(
        opts, trainset_augmented, valset, testset)
    train_loader = cifar10_original.train_loader
    train_loader_aug = cifar10_augmented.train_loader

    # Check first batch
    X_orig, y_orig = next(iter(train_loader))
    X_aug, y_aug = next(iter(train_loader_aug))
    print("Data shape:", f"X={X_orig.shape}", f"y={y_orig.shape}")

    # Print first batch
    import os
    os.makedirs("plots", exist_ok=True)
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    grid = make_grid(X_orig, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("plots/cifar10.png")

    grid = make_grid(X_aug, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("plots/cifar10_augmented.png")


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    from types import SimpleNamespace

    config = dict(seed=42, val_size=0.1,
                  batch_size=64, num_workers=2)
    opts = SimpleNamespace(**config)
    set_seeds(opts.seed)

    with launch_ipdb_on_exception():
        main_mnist(opts)
        print()
        main_cifar10(opts)
