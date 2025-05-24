"""Train the AutoEncoder"""

import time
from tqdm import tqdm

import torch

from lab1.utils import LOG
from lab1.utils.train_utils import N, save_checkpoint, AverageMeter


def train_loop(opts, model, optimizer, train_loader):
    """Simple training loop for the AutoEncoder"""
    step = 0
    start_time = time.time()

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
            X = X.to(opts.device)
            # forward pass
            z, X_r = model(X)  # outputs latent and reconstruction
            loss = criterion(X, X_r)
            # metrics
            losses.update(N(loss). X.size(0))
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
