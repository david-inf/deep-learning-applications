"""Training utilities for finetuning"""

import os
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn

from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from utils import N, LOG, AverageMeter, update_yaml


def save_model(opts, model: PreTrainedModel, reached_epoch, fname=None):
    """Save a pretrained model"""
    if not fname:
        fname = f"e_{reached_epoch:02d}_{opts.experiment_name}"
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    output_path = os.path.join(opts.checkpoint_dir, fname)
    model.save_pretrained(output_path)
    update_yaml(opts, "resume_checkpoint", output_path)

    LOG.info(f"Saved model at resume_checkpoint={opts.resume_checkpoint}")


def test(opts, model: PreTrainedModel, loader):
    """
    Evaluate model on test/validation set
    Loader can be either test_loader or val_loader
    """
    losses, accs = AverageMeter(), AverageMeter()
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()  # scalar value
    with torch.no_grad():
        for batch in loader:

            # Get data
            input_ids = batch["input_ids"].to(opts.device)
            attention_mask = batch["attention_mask"].to(opts.device)
            y = batch["labels"].to(opts.device)

            # Forward pass
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(output.logits, y)
            # Metrics
            losses.update(N(loss), input_ids.size(0))
            acc = np.mean(np.argmax(N(output.logits), axis=1) == N(y))
            accs.update(acc, input_ids.size(0))

    return losses.avg, accs.avg


# TODO: early stopping class


def train_epoch(opts, model: PreTrainedModel, optimizer, scheduler, train_loader, val_loader, epoch, step, experiment):
    """Train for a single epoch"""
    criterion = torch.nn.CrossEntropyLoss()
    losses, accs = AverageMeter(), AverageMeter()

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            model.train()
            tepoch.set_description(f"{epoch:03d}")

            # Get data
            input_ids = batch["input_ids"].to(opts.device)
            attention_mask = batch["attention_mask"].to(opts.device)
            y = batch["labels"].to(opts.device)

            # Forward pass
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(output.logits, y)
            # Metrics
            losses.update(N(loss), input_ids.size(0))
            acc = np.mean(np.argmax(N(output.logits), axis=1) == N(y))
            accs.update(acc, input_ids.size(0))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

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
                                   val_loss=val_loss, val_acc=val_acc
                                   )
                tepoch.update()
                step += 1

    return step


def train_loop(opts, model: PreTrainedModel, train_loader, val_loader, experiment):
    """Training loop for training a pretrained model with given finetuning setting"""
    cudnn.benchmark = True

    optimizer = optim.AdamW(model.parameters(), lr=opts.learning_rate)

    _total_steps = len(train_loader) * opts.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=_total_steps)

    start_epoch, step = 1, 0
    start_time = time.time()

    for epoch in range(start_epoch, opts.num_epochs + 1):
        experiment.log_current_epoch(epoch)

        step = train_epoch(opts, model, optimizer, scheduler, train_loader,
                           val_loader, epoch, step, experiment)

        # TODO: early stopping
        # TODO: checkpointing

    runtime = time.time() - start_time
    LOG.info(f"Training completed with runtime={runtime:.2f}s, "
             f"ended at epoch={epoch}, step={step}")
    experiment.log_metric("runtime", runtime)
    save_model(opts, model, epoch)
