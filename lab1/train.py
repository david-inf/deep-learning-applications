
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from utils import N, LOG, update_yaml


def save_checkpoint(opts, model, optimizer, scheduler, epoch, step, loss, runtime):
    """ Save a model checkpoint so training can be resumed and also wandb logging """
    fname = os.path.join(
        opts.checkpoint_dir,
        f"e_{epoch:03d}_{opts.experiment_name}.pt"
    )
    info = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        epoch=epoch,  # last epoch
        step=step,  # last step
        loss=loss,  # last computed loss
        runtime=runtime,  # duration of the run
    )
    torch.save(info, fname)
    # Update yaml file with checkpoint name
    update_yaml(opts, "resume_checkpoint", fname)
    LOG.info(f"Saved checkpoint {fname} at epoch {epoch}, "
             f"step {step}, runtime {runtime:.2f}s")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """ Load a model checkpoint to resume training """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}")

    # load from given checkpoint path
    LOG.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # load weights and optimizer in those given
    # this means that the initialized model and optimizer are updated
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)  # last completed epoch
    step = checkpoint.get("step", 0)  # last logged step
    loss = checkpoint.get("loss", float("inf"))  # loss at checkpoint
    runtime = checkpoint.get("runtime", 0.)

    # print(f"Resuming from epoch {epoch}, step {step}")
    return epoch, step, loss, runtime


def test(opts, model, test_loader):
    """ Evaluate model on test set """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    losses, correct = [], []
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for (X, y) in tepoch:
                tepoch.set_description("Test")

                X, y = X.to(opts.device), y.to(opts.device)
                out = model(X)  # logits: [N, K]
                # Compute loss
                loss = criterion(out, y)  # [N]
                losses.extend(N(loss))
                # Compute accuracy
                pred = np.argmax(N(out), axis=1)  # array of ints, size [N]
                label = N(y)  # {0,...,9}, size [N]
                c = list(pred == label)  # corrects [0,1,0,0,0,1,1...]
                correct.extend(c)

    # Compute mean loss and accuracy over the full test set
    return np.mean(correct)


def validate(opts, model, val_loader):
    """ Evaluate model on validation """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    losses, correct = [], []
    with torch.no_grad():
        for (X, y) in val_loader:

            X, y = X.to(opts.device), y.to(opts.device)
            out = model(X)  # logits: [N, K]
            # Compute loss
            loss = criterion(out, y)  # [N]
            losses.extend(N(loss))
            # Compute accuracy
            pred = np.argmax(N(out), axis=1)  # array of ints, size [N]
            label = N(y)  # {0,...,9}, size [N]
            c = list(pred == label)  # corrects [0,1,0,0,0,1,1...]
            correct.extend(c)

    val_loss = np.mean(losses)
    val_acc = np.mean(correct)

    # Compute mean loss and accuracy over the full test set
    return val_loss, val_acc


def train_loop(opts, model, train_loader, val_loader=None, experiment=None, resume_from=None):
    """
    Training loop with with resuming routine. This accounts for training
    ended before the number of epochs is reached or when one wants
    to train the model further.
    """
    criterion = torch.nn.CrossEntropyLoss()  # expects logits and labels
    optimizer = optim.SGD(model.parameters(), lr=opts.learning_rate,
                          momentum=opts.momentum, weight_decay=opts.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=opts.lr_decay)

    start_epoch, step = 1, 0  # last training epoch and step
    start_time, prev_runtime = time.time(), 0.

    if resume_from:
        # load checkpoint
        last_epoch, last_step, _, prev_runtime = load_checkpoint(
            resume_from, model, optimizer, scheduler)
        start_epoch += last_epoch
        step += last_step
        LOG.info(f"Resuming training from epoch {start_epoch}, step {step},"
                 f" previous runtime {prev_runtime:.2f}s")

    for epoch in range(start_epoch, opts.num_epochs + 1):
        experiment.log_current_epoch(epoch)
        losses, accs = [], []
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (X, y) in enumerate(tepoch):
                model.train()
                tepoch.set_description(f"{epoch:03d}")

                # -----
                # move data to device
                X = X.to(opts.device)  # [N, C, W, H]
                y = y.to(opts.device)  # [N]
                # forward pass
                optimizer.zero_grad()
                out = model(X)  # logits: [N, K]
                loss = criterion(out, y)  # scalar value
                # backward pass
                loss.backward()   # backprop
                optimizer.step()  # update model
                # metrics
                losses.append(N(loss))  # add loss for current batch
                acc = np.mean(np.argmax(N(out), axis=1) == N(y))
                accs.append(acc)  # add accuracy for current batch
                # -----

                if batch_idx % opts.log_every == 0:
                    # Compute training metrics and log to comet_ml
                    train_loss = np.mean(losses[-opts.batch_window:])
                    train_acc = np.mean(accs[-opts.batch_window:])
                    experiment.log_metrics({
                        "loss": train_loss,
                        "acc": train_acc,
                    }, step=step)
                    # Compute validation metrics and log to comet_ml
                    with experiment.validate():
                        val_loss, val_acc = validate(opts, model, val_loader)
                        experiment.log_metrics({
                            "loss": val_loss,
                            "acc": val_acc,
                        }, step=step)
                    # Log to console
                    tepoch.set_postfix(train_loss=train_loss, train_acc=train_acc,
                                       val_loss=val_loss, val_acc=val_acc)
                    tepoch.update()
                    step += 1

        scheduler.step()  # update learning rate at each epoch

        if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
            # save every checkpoint_every epochs and at the end
            ckp_runtime = prev_runtime + time.time() - start_time  # add duration of this run
            save_checkpoint(opts, model, optimizer, scheduler,
                            epoch, step, loss, ckp_runtime)

    # add this run duration to the previous one
    runtime = time.time() - start_time
    LOG.info(f"Training completed in {runtime:.2f}s")
    prev_runtime += runtime
    experiment.log_metric("runtime", prev_runtime)


def train_loop_distill(opts, teacher, student, train_loader, val_loader=None, experiment=None, resume_from=None):
    """
    Training loop for distillation

    Args:
        Teacher : torch.nn.Module
            teacher model with frozen parameters
        Student : torch.nn.Module
            student model to be distilled
    """
    ce_loss = nn.CrossEntropyLoss()
    kl_div = nn.KLDivLoss(log_target=True)
    optimizer = optim.SGD(student.parameters(), lr=opts.learning_rate,
                          momentum=opts.momentum, weight_decay=opts.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=opts.lr_decay)

    start_epoch, step = 1, 0  # last training epoch and step
    start_time, prev_runtime = time.time(), 0.

    for epoch in range(start_epoch, opts.num_epochs + 1):
        # experiment.log_current_epoch(epoch)
        losses, accs = [], []
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (X, y) in enumerate(tepoch):
                student.train()
                tepoch.set_description(f"{epoch:03d}")

                # -----
                # move data to device
                X = X.to(opts.device)  # [N, C, W, H]
                y = y.to(opts.device)  # hard targets [N]
                # forward pass
                optimizer.zero_grad()
                s_logits = student(X)  # student logits: [N, K]
                with torch.no_grad():
                    t_logits = teacher(X)  # teacher logits: [N, K]
                # compute loss
                soft_targets = F.log_softmax(t_logits / opts.temp, dim=-1)
                soft_prob = F.log_softmax(s_logits / opts.temp, dim=-1)
                loss1 = kl_div(soft_prob, soft_targets)  # soft targets loss
                loss2 = ce_loss(s_logits, y)
                loss = opts.w1 * loss1 + opts.w2 * loss2
                # backward pass
                loss.backward()   # backprop
                optimizer.step()  # update model
                # metrics
                losses.append(N(loss))  # add loss for current batch
                acc = np.mean(np.argmax(N(s_logits), axis=1) == N(y))
                accs.append(acc)  # add accuracy for current batch
                # -----

                if batch_idx % opts.log_every == 0:
                    # Compute training metrics and log to comet_ml
                    train_loss = np.mean(losses[-opts.batch_window:])
                    train_acc = np.mean(accs[-opts.batch_window:])
                    # experiment.log_metrics({
                    #     "loss": train_loss,
                    #     "acc": train_acc,
                    # }, step=step)
                    # Compute validation metrics and log to comet_ml
                    # with experiment.validate():
                    #     val_loss, val_acc = validate(opts, student, val_loader)
                    #     experiment.log_metrics({
                    #         "loss": val_loss,
                    #         "acc": val_acc,
                    #     }, step=step)
                    # Log to console
                    tepoch.set_postfix(train_loss=train_loss, train_acc=train_acc,
                                       #    val_loss=val_loss, val_acc=val_acc
                                       )
                    tepoch.update()
                    step += 1

        scheduler.step()  # update learning rate at each epoch

    # add this run duration to the previous one
    runtime = time.time() - start_time
    LOG.info(f"Training completed in {runtime:.2f}s")
    prev_runtime += runtime
    experiment.log_metric("runtime", prev_runtime)
