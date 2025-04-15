
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from utils import LOG, N, AverageMeter
from train import TrainState, test, load_checkpoint, save_checkpoint


def train_loop_distill(opts, teacher, student, train_loader, val_loader, experiment, resume_from=None):
    """
    Training loop for distillation

    Args:
        Teacher : torch.nn.Module
            teacher model with frozen parameters
        Student : torch.nn.Module
            student model to be distilled
    """
    cudnn.benchmark = True
    # loss for soft targets
    kl_div = nn.KLDivLoss(log_target=True, reduction="batchmean")
    # loss for hard targets
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student.parameters(), lr=opts.learning_rate,
                          momentum=opts.momentum, weight_decay=opts.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=opts.lr_decay)

    start_epoch, step = 1, 0  # last training epoch and step
    start_time, prev_runtime = time.time(), 0.

    if resume_from:
        # load checkpoint
        last_epoch, last_step, prev_runtime = load_checkpoint(
            resume_from, student, optimizer, scheduler)
        start_epoch += last_epoch
        step += last_step
        LOG.info(f"Resuming from epoch {start_epoch}, step {step},"
                 f" previous runtime {prev_runtime:.2f}s")

    trainer = TrainState(student, optimizer, scheduler,
                      start_epoch, step, prev_runtime)

    for epoch in range(start_epoch, opts.num_epochs + 1):
        # experiment.log_current_epoch(epoch)
        losses, accs = [], []
        with tqdm(train_loader, unit="batch") as tepoch:

            step = train_epoch(opts, teacher, student, val_loader, experiment,
                               ce_loss, kl_div, optimizer, step, epoch, losses,
                               accs, tepoch)

        scheduler.step()  # update learning rate at each epoch

        if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
            # save every checkpoint_every epochs or at the end
            ckp_runtime = prev_runtime + time.time() - start_time  # add duration of this run
            trainer.update(student, optimizer, scheduler,
                           epoch, step, ckp_runtime)
            save_checkpoint(trainer, opts)

    # add this run duration to the previous one
    runtime = time.time() - start_time
    LOG.info(f"Training completed in {runtime:.2f}s, "
             f"ended at epoch {epoch}, step {step}")
    prev_runtime += runtime
    LOG.info(f"Total runtime: {prev_runtime:.2f}s")
    experiment.log_metric("runtime", prev_runtime)


def train_epoch(opts, teacher, student, val_loader, experiment, ce_loss, kl_div, optimizer, step, epoch, tepoch):
    losses, accs = AverageMeter(), AverageMeter()
    for batch_idx, (X, y) in enumerate(tepoch):
        student.train()
        tepoch.set_description(f"{epoch:03d}")

        # -----
        # move data to device
        X = X.to(opts.device)  # [N, C, W, H]
        y = y.to(opts.device)  # hard targets [N]
        # forward pass
        s_logits = student(X)  # student logits: [N, K]
        t_logits = teacher(X)  # teacher logits: [N, K]
        # compute loss
        soft_targets = F.log_softmax(t_logits / opts.temp, dim=-1)
        soft_prob = F.log_softmax(s_logits / opts.temp, dim=-1)
        loss1 = kl_div(soft_prob, soft_targets)  # soft targets loss
        loss2 = ce_loss(s_logits, y)
        loss = opts.w1 * loss1 + opts.w2 * loss2
        # metrics
        losses.update(N(loss), X.size(0))
        acc = np.mean(np.argmax(N(s_logits), axis=1) == N(y))
        accs.update(acc, X.size(0))
        # backward pass
        optimizer.zero_grad()
        loss.backward()   # backprop
        optimizer.step()  # update model
        # -----

        if batch_idx % opts.log_every == 0:
            # Compute training metrics and log to comet_ml
            # train_loss = np.mean(losses)  # [-opts.batch_window:])
            # train_acc = np.mean(accs)  # [-opts.batch_window:])
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
