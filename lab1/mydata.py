
import random
import numpy as np

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class MyMNIST(Dataset):
    """ Wrapper for MNIST Dataset class """

    def __init__(self, opts):
        # TODO: add data augmentation (maybe another class)
        self.opts = opts

        # Get fill MNIST dataset
        # (X, y) train: N=60000 ; test: N=10000
        trainset = datasets.MNIST(
            root="../../data", train=True, download=True)
        testset = datasets.MNIST(
            root="../../data", train=False, download=True)

        # Combine train and test sets
        X_train, y_train = trainset.data, trainset.targets
        X_test, y_test = testset.data, testset.targets

        # All images and all targets
        X = np.vstack((X_train, X_test))  # [70000, 28, 28]
        y = np.hstack((y_train, y_test))  # [70000]

        # Prepare data
        self.X = torch.tensor(X).unsqueeze(
            1) / 255.0  # [N, 1, 28, 28] in [0,1]
        self.y = torch.tensor(y)  # [N]

        # Normalize data over channels
        self.mean = self.X.mean()
        self.std = self.X.std()
        self.X = transforms.Normalize(self.mean, self.std)(self.X)

    def __len__(self):
        return self.X.shape[0]  # 70000

    def __getitem__(self, idx):
        """
        returns:
            X: [C, W, H] (tensor)
            y: [] (tensor)
        """
        return self.X[idx], self.y[idx]


class MyCIFAR10(Dataset):
    """ Wrapper for CIFAR10 Dataset class """

    def __init__(self, opts=None, crop=True):
        self.opts = opts

        # Get full CIFAR10 dataset, first separated
        # (X, y) train: N=50000 ; test: N=10000
        trainset = datasets.CIFAR10(
            root="../../data", train=True, download=True)
        testset = datasets.CIFAR10(
            root="../../data", train=False, download=True)
        self.num_classes = len(CIFAR10_CLASSES)

        # Combine train and test sets
        # so that corruption can be applied to both at the same time
        # PIL Image <> numpy array ; int <> list
        X_train, y_train = trainset.data, trainset.targets
        X_test, y_test = testset.data, testset.targets

        X = np.vstack((X_train, X_test))  # [60000, 32, 32, 3]
        y = np.hstack((y_train, y_test))  # [60000]

        # Prepare data
        self.X = torch.tensor(X).permute(0, 3, 1, 2) / 255.0  # scale to [0,1]
        if crop:  # take central part of each image
            margin = (32 - 28) // 2  # 28x28 image
            self.X = self.X[:, :, margin:-margin, margin:-margin]
        self.y = torch.tensor(y)

        # Data normalization
        self.mean = torch.mean(self.X, dim=(0, 2, 3))  # original data mean
        self.std = torch.std(self.X, dim=(0, 2, 3))  # original data std
        for c in range(3):  # For each channel
            self.X[:, c, :, :] = (self.X[:, c, :, :] -
                                  self.mean[c]) / self.std[c]

    def __len__(self):
        return self.X.shape[0]  # 60000

    def __getitem__(self, idx):
        """
        returns:
            X: [C, W, H] (tensor)
            y: [] (tensor)
        """
        return self.X[idx], self.y[idx]


class MakeDataLoaders():
    def __init__(self, opts, data):
        generator = torch.Generator().manual_seed(opts.seed)
        train_full, test = random_split(
            data, lengths=[1-opts.test_size, opts.test_size],
            generator=generator
        )
        train, val = random_split(
            train_full, lengths=[1-opts.val_size, opts.val_size],
            generator=generator
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.train_loader = DataLoader(
            train, batch_size=opts.batch_size, shuffle=True,
            num_workers=opts.num_workers, pin_memory=True,
            generator=generator, worker_init_fn=seed_worker
        )
        self.val_loader = DataLoader(
            val, batch_size=opts.batch_size, shuffle=True,
            num_workers=opts.num_workers, pin_memory=True,
            generator=generator, worker_init_fn=seed_worker
        )
        self.test_loader = DataLoader(
            test, batch_size=opts.batch_size, shuffle=True,
            num_workers=opts.num_workers, pin_memory=True,
            generator=generator, worker_init_fn=seed_worker
        )


def main_mnist(opts):
    data = MyMNIST(opts)
    print(f"Dataset size: X={data.X.shape}, y={data.y.shape}")
    print(f"Mean: {data.mean:.4f}, Std: {data.std:.4f}")
    mnist = MakeDataLoaders(opts, data)
    train_loader = mnist.train_loader

    # Check data
    X, y = next(iter(train_loader))
    print("Data shape:", f"X={X.shape}", f"y={y.shape}")

    # Print first batch
    import os
    os.makedirs("plots", exist_ok=True)
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    grid = make_grid(X, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("plots/mnist.png")


def main_cifar10(opts):
    data = MyCIFAR10(opts)
    print(f"Dataset size: X={data.X.shape}, y={data.y.shape}")
    print(f"Mean: {data.mean}, Std: {data.std}")
    cifar10 = MakeDataLoaders(opts, data)
    train_loader = cifar10.train_loader

    # Check data
    X, y = next(iter(train_loader))
    print("Data shape:", f"X={X.shape}", f"y={y.shape}")

    # Print first batch
    import os
    os.makedirs("plots", exist_ok=True)
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    grid = make_grid(X, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("plots/cifar10.png")


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    from types import SimpleNamespace

    config = dict(seed=42, test_size=0.2, val_size=0.1,
                  batch_size=64, num_workers=2)
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        print("MNIST")
        main_mnist(opts)
        print("\nCIFAR10")
        main_cifar10(opts)
