""" Training utilities """

import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.backends.cudnn as cudnn

from utils import N, LOG, update_yaml, AverageMeter


def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler):
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
    runtime = checkpoint.get("runtime", 0.)

    # print(f"Resuming from epoch {epoch}, step {step}")
    return epoch, step, runtime


def test(opts, model, loader):
    """
    Evaluate model on test/validation set
    Loader can be either test_loader or val_loader
    """
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()  # scalar value
    with torch.no_grad():
        for (X, y) in loader:

            # Get data and forward pass
            X, y = X.to(opts.device), y.to(opts.device)
            out = model(X)  # logits: [N, K]
            # Compute loss
            loss = criterion(out, y)
            losses.update(N(loss), X.size(0))
            # Compute accuracy
            acc = np.mean(np.argmax(N(out), axis=1) == N(y))
            accs.update(acc, X.size(0))

    return losses.avg, accs.avg


class TrainState:
    """ Class to store training state """

    def __init__(self, model, optimizer, scheduler, epoch, step, runtime):
        self.model = model.state_dict()
        self.optimizer = optimizer.state_dict()
        self.scheduler = scheduler.state_dict()
        self.epoch = epoch  # last epoch reached
        self.step = step  # last step reached
        self.runtime = runtime  # total runtime

    def update(self, model, optimizer, scheduler, epoch, step, runtime):
        self.model = model.state_dict()
        self.optimizer = optimizer.state_dict()
        self.scheduler = scheduler.state_dict()
        self.epoch = epoch
        self.step = step
        self.runtime = runtime

    def __dict__(self):
        """
        Return a dictionary representation of the training state
        that will be used for checkpointing
        """
        return {"model_state_dict": self.model,
                "optimizer_state_dict": self.optimizer,
                "scheduler_state_dict": self.scheduler, "epoch": self.epoch,
                "step": self.step, "runtime": self.runtime, }


def save_checkpoint(trainer: TrainState, opts, fname=None):
    """ Save a model checkpoint to be resumed later """
    info = trainer.__dict__()
    if not fname:
        fname = f"e_{info["epoch"]:03d}_{opts.experiment_name}.pt"
    output_dir = os.path.join(opts.checkpoint_dir, fname)
    torch.save(info, output_dir)
    # Update yaml file with checkpoint name
    update_yaml(opts, "resume_checkpoint", output_dir)
    LOG.info(f"Saved checkpoint {fname} at epoch {info["epoch"]}, "
             f"step {info["step"]}, runtime {info["runtime"]:.2f}s")


class EarlyStopping:
    """
    Early stopping strategy
    One of the best forms of explicit regularization
    """

    def __init__(self, configs: dict):
        self.patience = configs["patience"]  # steps to wait
        self.thresh = configs["threshold"]
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_trainer = None

    def __call__(self, val_acc, trainer: TrainState):
        score = val_acc
        if self.best_score is None:
            # initialize best score
            self.best_score = score
            self.best_trainer = trainer
        elif score < self.best_score + self.thresh:
            # no improvements
            self.counter += 1
            if self.counter >= self.patience:
                # stop training when we see no improvements
                # for patience number of epochs
                self.early_stop = True
                # at this point we should stop training
                # and save checkpoint
        else:
            # we see an improvement
            self.best_score = score
            self.best_trainer = trainer
            self.counter = 0


def train_loop(opts, model, train_loader, val_loader, experiment, resume_from):
    """
    Training loop with resuming routine. This accounts for training
    ended before the number of epochs is reached or when one wants
    to train the model further.

    Args:
        opts: Configuration options for training.
        model: The model to be trained.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        experiment: Experiment object for logging.
        resume_from: Path to a checkpoint to resume training from.
    """
    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss()  # expects logits and labels
    optimizer = optim.SGD(model.parameters(), lr=opts.learning_rate,
                          momentum=opts.momentum, weight_decay=opts.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=opts.lr_decay)

    start_epoch, step = 1, 0  # last training epoch and step
    start_time, prev_runtime = time.time(), 0.

    if resume_from:
        # load checkpoint
        last_epoch, last_step, prev_runtime = load_checkpoint(
            resume_from, model, optimizer, scheduler)
        start_epoch += last_epoch
        step += last_step
        LOG.info(f"Resuming from epoch {start_epoch}, step {step},"
                 f" previous runtime {prev_runtime:.2f}s")

    # save training objects and info
    trainer = TrainState(model, optimizer, scheduler,
                      start_epoch, step, prev_runtime)
    # keeps the training objects and info from the best model
    if hasattr(opts, "early_stopping") and opts.early_stopping:
        early_stopping = EarlyStopping(opts.early_stopping)

    for epoch in range(start_epoch, opts.num_epochs + 1):
        experiment.log_current_epoch(epoch)

        step, val_acc = train_epoch(
            opts, model, train_loader, val_loader, experiment, criterion,
            optimizer, step, epoch)

        scheduler.step()  # update learning rate at each epoch

        if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
            # save every checkpoint_every epochs or at the end
            ckp_runtime = prev_runtime + time.time() - start_time  # add duration of this run
            trainer.update(model, optimizer, scheduler,
                           epoch, step, ckp_runtime)
            save_checkpoint(trainer, opts)

        # Check for early stopping after each epoch
        if hasattr(opts, "early_stopping"):
            estop_runtime = prev_runtime + time.time() - start_time
            trainer.update(model, optimizer, scheduler,
                           epoch, step, estop_runtime)
            early_stopping(val_acc, trainer)  # use last computed val_acc
            if early_stopping.early_stop:
                # Do early stopping
                fname = f"e_{epoch:03d}_{opts.experiment_name}_best.pt"
                save_checkpoint(early_stopping.best_trainer, opts, fname)
                LOG.info(f"Early stopping at epoch {epoch}, "
                         f"step {step}, checkpoint {fname}")
                break

    # add this run duration to the previous one
    runtime = time.time() - start_time
    LOG.info(f"Training completed in {runtime:.2f}s, "
             f"ended at epoch {epoch}, step {step}")
    prev_runtime += runtime
    LOG.info(f"Total runtime: {prev_runtime:.2f}s")
    experiment.log_metric("runtime", prev_runtime)


def train_epoch(opts, model, train_loader, val_loader, experiment, criterion, optimizer, step, epoch):
    """Train over a single epoch"""
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
            acc = np.mean(np.argmax(N(out), axis=1) == N(y))
            accs.update(acc, X.size(0))
            # backward pass
            optimizer.zero_grad()
            loss.backward()   # backprop
            optimizer.step()  # update model
            # -----

            if batch_idx % opts.log_every == 0:
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
