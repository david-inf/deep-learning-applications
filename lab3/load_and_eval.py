"""Load model from checkpoint and evaluate on testset"""

import os

from accelerate import Accelerator
from torch.backends import cudnn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from main_ft import get_loaders
from utils import LOG
from train import test


def get_model(opts):
    """Load model from checkpoint"""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        opts.checkpoint, num_labels=2)
    return tokenizer, model


def main(opts):
    """Load model from checkpoint, then evaluate on testset"""
    # Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    LOG.info("Accelerator device=%s", accelerator.device)

    # Load finetuned model
    tokenizer, model = get_model(opts)
    # Load test set
    with accelerator.main_process_first():
        if opts.split == "validation":
            _, loader, _ = get_loaders(opts, tokenizer)
        elif opts.split == "test":
            _, _, loader = get_loaders(opts, tokenizer)
        else:
            raise ValueError(f"Unknown split {opts.split}")

    # Prepare for evaluation
    cudnn.benchmark = True
    model, loader = accelerator.prepare(
        model, loader)

    # Evaluate
    _, val_acc = test(model, accelerator, loader)
    LOG.info("val_acc=%.3f", val_acc)


if __name__ == "__main__":
    import argparse
    import yaml
    from types import SimpleNamespace

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--split", default="test", help="RT dataset split")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        opts = yaml.safe_load(f)
    # Update opts with command line arguments
    opts.update(vars(args))
    opts = SimpleNamespace(**opts)

    # Check if the checkpoint exists
    if not os.path.exists(opts.checkpoint):
        raise FileNotFoundError(f"Checkpoint {opts.checkpoint} does not exist")
    try:
        main(opts)
    except Exception as e:
        print(e)
