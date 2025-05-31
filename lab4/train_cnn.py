"""Training utilities"""

import sys
import os
import time
import random

import torch
import numpy as np
from tqdm import tqdm

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from lab1.utils.train import N, LOG, AverageMeter, save_checkpoint, accuracy, EarlyStopping
from lab1.train import test

from lab4.main_adversarial import attack


def train_loop(opts, model, optimizer, scheduler, train_loader, val_loader):
    """Training loop for adversarial learning"""
    step = 0
    start_time = time.time()

    if opts.do_early_stopping:
        early_stopping = EarlyStopping(opts)

    save_checkpoint(opts, model)
    for epoch in range(1, opts.num_epochs + 1):

        step, val_acc = train_epoch(
            opts, model, optimizer, train_loader, val_loader, step, epoch)
        scheduler.step()  # update learning rate at each epoch

        if opts.do_early_stopping:
            # check for early stopping after each epoch
            early_stopping(val_acc, model)
            if early_stopping.early_stop:
                LOG.info("Early stopping triggered, breaking training...")
                LOG.info("val_acc=%.3f <> best_val_acc=%.3f",
                         val_acc, early_stopping.best_score)
                break
        else:
            if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
                # save checkpoint every opts.checkpoint_every epochs
                save_checkpoint(opts, model)

    runtime = time.time() - start_time
    LOG.info("Training completed in %.2fs, ended at epoch %d, step %d",
             runtime, epoch, step)


def train_epoch(opts, model, optimizer, train_loader, val_loader, step, epoch):
    """Train over a single epoch"""
    criterion = torch.nn.CrossEntropyLoss()  # expects logits and labels
    losses, accs = AverageMeter(), AverageMeter()
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (X, y) in enumerate(tepoch):
            model.train()
            tepoch.set_description(f"{epoch:03d}")

            # move data to device
            X = X.to(opts.device)  # [N, C, W, H]
            y = y.to(opts.device)  # [N]
            # forward pass
            out = model(X)  # logits: [N, K]

            # adversarial augmentations
            # the idea is to corrupt existing training samples
            # opts.adversarial: dict with adv params
            preds = np.argmax(N(out), axis=1)
            X = adversarial_augmentations(opts, model, X, y, preds)

            # forward pass with adversarial examples
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
                # Compute validation metrics and log to comet_ml
                val_loss, val_acc = test(opts, model, val_loader)
                # Log to console
                tepoch.set_postfix(train_loss=train_loss, train_acc=train_acc,
                                   val_loss=val_loss, val_acc=val_acc)
                tepoch.update()
                step += 1

    return step, val_acc


def adversarial_augmentations(opts, model, images, labels, preds):
    """Perform adversarial attacks"""
    success_count = 0

    for i in range(images.size(0)):
        if random.random() < opts.fraction:
            if preds[i] != labels[i].item():
                # classifier is wrong
                continue

            # attack when classifier is correct
            # so that the classifier can get robust to attacks
            image = images[i].clone()
            image.requires_grad = True
            image, _, iters = attack(
                labels[i], image,model, opts.budget/255)
            image = image.detach()
            images[i] = image

            if iters > 0:
                success_count += 1

    return images
