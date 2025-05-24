"""Training utilities"""

import time
from tqdm import tqdm

from comet_ml import Experiment
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lab1.utils.misc import LOG
from lab1.utils.train import N, accuracy, save_checkpoint, AverageMeter, EarlyStopping


def test(opts, model, loader):
    """Evaluate model on validation set"""
    criterion = torch.nn.CrossEntropyLoss()  # scalar value
    losses, accs = AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        for (X, y) in loader:
            # Get data and forward pass
            X, y = X.to(opts.device), y.to(opts.device)
            out = model(X)  # logits: [N, K]
            # Compute loss
            loss = criterion(out, y)
            losses.update(N(loss), X.size(0))
            # Compute accuracy
            acc = accuracy(N(out), N(y))
            accs.update(acc, X.size(0))

    return losses.avg, accs.avg


def train_loop(opts, model: Module, optimizer: Optimizer, scheduler: LRScheduler, train_loader, val_loader, experiment: Experiment):
    """
    Training loop with resuming routine. This accounts for training
    ended before the number of epochs is reached or when one wants
    to train the model further.
    """
    start_epoch, step = 1, 0
    start_time = time.time()

    # keeps the training objects and info from the best model
    if opts.do_early_stopping:
        early_stopping = EarlyStopping(opts)

    for epoch in range(start_epoch, opts.num_epochs + 1):
        experiment.log_current_epoch(epoch)

        step, val_acc = train_epoch(
            opts, model, optimizer, train_loader, val_loader, experiment, step, epoch)
        scheduler.step()  # update learning rate at each epoch

        # Check for early stopping after each epoch
        if opts.do_early_stopping:
            # TODO: add _best in checkpoint name?
            early_stopping(val_acc, model)  # use last computed val_acc
            if early_stopping.early_stop:
                LOG.info("Early stopping triggered, breaking training...")
                LOG.info("val_acc=%.3f <> best_val_acc=%.3f",
                         val_acc, early_stopping.best_score)
                break
        else:
            if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
                # save every checkpoint_every epochs or at the end
                save_checkpoint(opts, model)

    # add this run duration to the previous one
    runtime = time.time() - start_time
    LOG.info("Training completed in %.2fs, ended at epoch %d, step %d",
             runtime, epoch, step)
    experiment.log_metric("runtime", runtime)


def train_epoch(opts, model: Module, optimizer: Optimizer, train_loader, val_loader, experiment: Experiment, step, epoch):
    """Train over a single epoch"""
    criterion = torch.nn.CrossEntropyLoss()  # expects logits and labels
    losses, accs = AverageMeter(), AverageMeter()
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (X, y) in enumerate(tepoch):
            model.train()
            tepoch.set_description(f"{epoch:03d}")

            # -----
            # move data to device
            X = X.to(opts.device)  # [N, C, W, H]
            y = y.to(opts.device)  # [N]
            # forward pass
            out = model(X)  # logits: [N, K]
            loss = criterion(out, y)  # scalar value
            # metrics
            losses.update(N(loss), X.size(0))
            acc = accuracy(N(out), N(y))
            accs.update(acc, X.size(0))
            # backward pass
            optimizer.zero_grad()
            loss.backward()   # backprop
            optimizer.step()  # update model
            # -----

            if batch_idx % opts.log_every == 0 or batch_idx == len(train_loader) - 1:
                # Compute training metrics and log to comet_ml
                train_loss, train_acc = losses.avg, accs.avg
                experiment.log_metrics(
                    {"loss": train_loss, "acc": train_acc}, step=step)
                # Compute validation metrics and log to comet_ml
                with experiment.validate():
                    val_loss, val_acc = test(opts, model, val_loader)
                    experiment.log_metrics(
                        {"loss": val_loss, "acc": val_acc}, step=step)
                # Log to console
                tepoch.set_postfix(train_loss=train_loss, train_acc=train_acc,
                                   val_loss=val_loss, val_acc=val_acc)
                tepoch.update()
                step += 1

    return step, val_acc
