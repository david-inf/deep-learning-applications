"""Train the AutoEncoder"""

import sys
import os
import time
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.backends import cudnn

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from lab1.utils import LOG, set_seeds
from lab1.utils.train import N, AverageMeter, save_checkpoint
from lab1.utils.misc import visualize
from lab4.mydata import get_loaders
from lab4.models import AutoEncoder


def train_loop(opts, model, optimizer, train_loader):
    """Simple training loop for the AutoEncoder"""
    step = 0
    start_time = time.time()

    save_checkpoint(opts, model)
    for epoch in range(1, opts.num_epochs + 1):

        step = train_epoch(opts, model, optimizer, train_loader, step, epoch)

        if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
            save_checkpoint(opts, model)
    
    runtime = time.time() - start_time
    LOG.info("Training completed in %.2fs, ended at epoch %d, step %d",
             runtime, epoch, step)


def train_epoch(opts, model, optimizer, train_loader, step, epoch):
    """Single epoch training"""
    criterion = torch.nn.MSELoss()
    losses = AverageMeter()
    with tqdm(train_loader, total=len(train_loader), unit="batch") as tepoch:
        for batch_idx, (X, _) in enumerate(tepoch):
            model.train()
            tepoch.set_description(f"{epoch:02d}")

            # move data to device
            X = X.to(opts.device)  # [N, 3, 28, 28]
            # forward pass
            X_r = model(X)  # outputs reconstruction
            loss = criterion(X, X_r)
            # metrics
            losses.update(N(loss), X.size(0))
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % opts.log_every == 0 or batch_idx == len(train_loader) - 1:
                # Compute training metrics and log to comet_ml
                train_loss = losses.avg
                # Log to console
                tepoch.set_postfix(train_loss=train_loss)
                tepoch.update()
                step += 1

    return step


def main(opts):
    """AutoEncoder training"""
    set_seeds(opts.seed)
    # Loaders
    # make sure the data is in [0,1]
    train_loader = get_loaders(opts, True)

    # Model and Optimizer
    model = AutoEncoder(opts.num_filters).to(opts.device)
    optimizer = Adam(model.parameters(), lr=opts.learning_rate)

    # Training
    cudnn.benchmark = True
    LOG.info("experiment_name=%s", opts.experiment_name)
    train_loop(opts, model, optimizer, train_loader)


def view_model(opts):
    """AutoEncoder inspection"""
    opts.device = "cpu"
    model = AutoEncoder().to(opts.device)
    input_data = torch.randn(opts.batch_size, 3, 28, 28)
    visualize(model, "AutoEncoder", input_data)


if __name__ == "__main__":
    import argparse
    import yaml
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(
        description="Train a AutoEncoder to reconstruct in-distribution images")
    parser.add_argument("--config", help="AutoEncoder YAML configuration file")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)  # dict
    opts = SimpleNamespace(**configs)

    try:
        if not args.view:
            main(opts)
        else:
            view_model(opts)
    except Exception:
        import ipdb, traceback
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
