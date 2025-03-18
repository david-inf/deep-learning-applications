
import random
import numpy as np

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


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
        self.X = torch.tensor(X).unsqueeze(1) / 255.0  # [N, 1, 28, 28] in [0,1]
        self.y = torch.tensor(y)  # [N]

        # Normalize data over channels
        self.mean = self.X.mean()
        self.std = self.X.std()
        self.X = transforms.Normalize(self.mean, self.std)(self.X)

    def __len__(self):
        return self.X.shape[0]

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


def main(opts):
    data = MyMNIST(opts)
    print(f"Dataset size: X={data.X.shape}, y={data.y.shape}")
    print(f"Mean: {data.mean}, Std: {data.std}")
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


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    from types import SimpleNamespace

    config = dict(seed=42, test_size=0.2, val_size=0.1,
                  batch_size=64, num_workers=0)
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
