"""Main program for finetuning BERT for sentiment analysis"""

import os

import torch
from torch.optim import AdamW
from torch.backends import cudnn
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup, PreTrainedModel

from mydata import get_loaders
from models.distilbert import get_distilbert
from utils.misc import LOG, visualize, update_yaml
from train import train_loop


def get_model(opts):
    """Get model to finetune"""
    if opts.model == "distilbert":
        tokenizer, model = get_distilbert(opts)
    else:
        raise ValueError(f"Unknown model {opts.model}")
    return tokenizer, model


def get_optimization(opts, model: PreTrainedModel, train_loader):
    """Optimizer and LRScheduler settings"""
    configs = opts.ft_setting  # dict
    head_params = [p for name, p in model.named_parameters()
                   if "classifier" in name]
    backbone_params = [
        p for name, p in model.named_parameters() if "classifier" not in name]
    params = [
        {"params": head_params, "lr": configs["lr_head"]},
        {"params": backbone_params},
    ]
    optimizer = AdamW(
        params,
        lr=opts.learning_rate,
        weight_decay=opts.weight_decay
    )

    total_steps = opts.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(opts.warmup*total_steps),
        total_steps
    )

    return optimizer, scheduler


def main(opts):
    set_seed(opts.seed)

    # Checkpointing
    os.makedirs(opts.checkpoint_dir, exist_ok=True)
    output_path = os.path.join(opts.checkpoint_dir, opts.experiment_name)
    update_yaml(opts, "checkpoint", output_path)

    # Accelerator
    accelerator = Accelerator(mixed_precision="fp16", project_dir=output_path)
    LOG.info("Accelerator device=%s", accelerator.device)

    # Get BERT and its tokenizer (cpu)
    tokenizer, model = get_model(opts)
    # Get loaders
    with accelerator.main_process_first():
        train_loader, val_loader, _ = get_loaders(opts, tokenizer)

    # Optimizer
    optimizer, scheduler = get_optimization(opts, model, train_loader)

    # Prepare training -> send to device
    cudnn.benchmark = True
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)
    accelerator.register_for_checkpointing(scheduler)

    # Training
    LOG.info("Running experiment_name=%s", opts.experiment_name)
    train_loop(opts, model, optimizer, scheduler,
               accelerator, train_loader, val_loader)


def view_model(opts):
    """DistilBERT inspection"""
    # Get BERT and tokenizer
    # TODO: sbert and other integrations
    tokenizer, model = get_model(opts)

    # Random data for input_ids and attention_mask
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (opts.batch_size, 128))
    attention_mask = torch.ones(
        (opts.batch_size, 128), dtype=torch.int64)
    input_data = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Visualize model
    visualize(model, opts.model, input_data)


if __name__ == "__main__":
    from cmd_args import parse_args
    args = parse_args()
    try:
        if args.visualize:
            view_model(args)
        else:
            main(args)
    except Exception:
        import ipdb
        import traceback
        import sys
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
