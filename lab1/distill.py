
import os
import sys
import time

from comet_ml import Experiment
import torch
from torch import nn

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F
from tqdm import tqdm

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from lab1.utils.misc import LOG
from lab1.utils.train import N, accuracy, save_checkpoint, AverageMeter
from lab1.train import test


def train_loop_distill(opts, teacher: Module, student: Module, optimizer: Optimizer, scheduler: LRScheduler, train_loader, val_loader, experiment: Experiment):
    """Training loop for distillation"""
    start_epoch, step = 1, 0
    start_time = time.time()

    for epoch in range(start_epoch, opts.num_epochs + 1):
        experiment.log_current_epoch(epoch)

        step = train_epoch(
            opts, teacher, student, optimizer, train_loader, val_loader,
            experiment, step, epoch)
        scheduler.step()  # update learning rate at each epoch

        if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
            # save every checkpoint_every epochs or at the end
            save_checkpoint(opts, student)

    # add this run duration to the previous one
    runtime = time.time() - start_time
    LOG.info("Training completed in %.2fs, ended at epoch %d, step %d",
             runtime, epoch, step)
    experiment.log_metric("runtime", runtime)


def train_epoch(opts, teacher: Module, student: Module, optimizer: Optimizer, train_loader, val_loader, experiment: Experiment, step, epoch):
    """Train over a single epoch"""
    kl_div = nn.KLDivLoss(log_target=True, reduction="batchmean")  # loss for soft targets
    ce_loss = nn.CrossEntropyLoss()  # loss for hard targets
    losses, accs = AverageMeter(), AverageMeter()
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (X, y) in enumerate(tepoch):
            student.train()
            tepoch.set_description(f"{epoch:03d}")

            # -----
            # move data to device
            X = X.to(opts.device)  # [N, C, W, H]
            y = y.to(opts.device)  # hard targets [N]
            # forward pass
            stud_out = student(X)  # student logits: [N, K]
            with torch.no_grad():  # double check on freezed params
                teach_out = teacher(X)  # teacher logits: [N, K]
            # compute loss
            labloss = ce_loss(stud_out, y)  # labels loss
            soft_targets = F.log_softmax(teach_out / opts.temp, dim=-1)
            soft_prob = F.log_softmax(stud_out / opts.temp, dim=-1)
            stloss = kl_div(soft_prob, soft_targets)  # soft targets loss
            loss = opts.weight_stloss * stloss + opts.weight_labloss * labloss
            # metrics
            losses.update(N(loss), X.size(0))
            acc = accuracy(N(stud_out), N(y))
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
                    val_loss, val_acc = test(opts, student, val_loader)
                    experiment.log_metrics(
                        {"loss": val_loss, "acc": val_acc}, step=step)
                # Log to console
                tepoch.set_postfix(train_loss=train_loss, train_acc=train_acc,
                                   val_loss=val_loss, val_acc=val_acc)
                tepoch.update()
                step += 1

    return step
