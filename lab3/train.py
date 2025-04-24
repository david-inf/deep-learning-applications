"""Training utilities for finetuning"""

import os
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn

from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from utils import N, LOG, AverageMeter


def save_model(opts, model: PreTrainedModel, reached_epoch, fname=None):
    """Save a pretrained model"""
    if not fname:
        fname = f"e_{reached_epoch:02d}_{opts.experiment_name}"
    output_dir = os.path.join(opts.checkpoint_dir, fname)
    model.save_pretrained(output_dir)
    LOG.info(f"Saved model at path={fname}")


def test(opts, model, loader):
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


def train_epoch(opts, model, optimizer, train_loader, val_loader, epoch, step, experiment):
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

            if batch_idx % opts.log_every == 0:
                # Compute training metrics and log to comet_ml
                train_loss, train_acc = losses.avg, accs.avg
                experiment.log_metrics(
                    {"loss": train_loss, "acc": train_acc}, step=step)
                # Compute validation metrics and log to comet_ml
                # with experiment.validate():
                #     val_loss, val_acc = test(opts, model, val_loader)
                #     experiment.log_metrics(
                #         {"loss": val_loss, "acc": val_acc}, step=step)
                # Log to console
                tepoch.set_postfix(train_loss=train_loss, train_acc=train_acc,
                                   #   val_loss=val_loss, val_acc=val_acc
                                   )
                tepoch.update()
                step += 1

    return step


def train_loop(opts, model, train_loader, val_loader, experiment):
    cudnn.benchmark = True
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=opts.learning_rate)
    # scheduler = get_linear_schedule_with_warmup(optimizer, )

    start_epoch, step = 1, 0
    start_time = time.time()

    for epoch in range(start_epoch, opts.num_epochs + 1):
        experiment.log_current_epoch(epoch)

        step = train_epoch(opts, model, optimizer, train_loader,
                           val_loader, epoch, step, experiment)

        # TODO: early stopping
        # TODO: checkpointing

    runtime = time.time() - start_time
    LOG.info(f"Training completed with runtime={runtime:.2f}s, "
             f"ended at epoch={epoch}, step={step}")
    experiment.log_metric("runtime", runtime)
