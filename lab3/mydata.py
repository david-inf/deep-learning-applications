
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from transformers import DataCollatorWithPadding
from datasets import load_dataset

from utils import set_seeds, LOG
from models.bert import get_bert


class MakeDataLoaders:
    def __init__(self, opts, tokenizer, trainset: Dataset, valset: Dataset, testset: Dataset):
        set_seeds(opts.seed)
        generator = torch.Generator().manual_seed(opts.seed)
        collate_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            # dynamic padding, different per each batch
            padding="longest"
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.train_loader = DataLoader(
            trainset, batch_size=opts.batch_size, num_workers=opts.num_workers,
            pin_memory=True, generator=generator, worker_init_fn=seed_worker,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            valset, batch_size=opts.batch_size, num_workers=opts.num_workers,
            pin_memory=True, generator=generator, worker_init_fn=seed_worker,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            testset, batch_size=opts.batch_size, num_workers=opts.num_workers,
            pin_memory=True, generator=generator, worker_init_fn=seed_worker,
            collate_fn=collate_fn,
        )


def get_loaders(opts, tokenizer):
    # 1) Get dataset splits
    if opts.dataset == "rotten_tomatoes":
        rt_trainset, rt_valset, rt_testset = load_dataset(
            "cornell-movie-review-data/rotten_tomatoes",
            split=["train", "validation", "test"])
    else:
        raise ValueError(f"Unknown dataset {opts.dataset}")

    # 2) Preprocess data
    def preprocess(sample):
        return tokenizer(
            # tokenize the text without padding
            sample["text"],
            # truncate to specified length if necessary
            max_length=opts.max_length,
            truncation=True,
            return_attention_mask=True,
            # returns lists as the default collator wants
            return_tensors=None,
        )

    trainset = rt_trainset.map(
        preprocess, batched=True, num_proc=2, remove_columns=["text"], desc="Tokenizing")
    valset = rt_valset.map(
        preprocess, batched=True, remove_columns=["text"], desc="Tokenizing")
    testset = rt_testset.map(
        preprocess, batched=True, remove_columns=["text"], desc="Tokenizing")

    # 3) Loaders
    loaders = MakeDataLoaders(opts, tokenizer, trainset, valset, testset)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    test_loader = loaders.test_loader

    return train_loader, val_loader, test_loader


def main(opts):
    # Get tokenizer
    tokenizer, _ = get_bert(opts)
    # Get loaders
    loaders = get_loaders(opts, tokenizer)

    # Inspect first batch
    sets = ["Train", "Val", "Test"]
    for i, loader in enumerate(loaders):
        # get batch
        batch = next(iter(loader))
        # get elements
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        labels = batch["labels"]
        # inspect
        LOG.info(f"{sets[i]}"
                 f"\ninput_ids={input_ids.shape}"
                 f"\nattention_mask={attn_mask.shape}"
                 f"\nlabels={labels.shape}")
        print()

    for batch_idx, batch in enumerate(loaders[0]):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        LOG.info(f"Train batch_idx={batch_idx}"
                 f"\ntokens={input_ids.shape} "
                 f"mask={attention_mask.shape} "
                 f"y={labels.shape}")

        if batch_idx == 1:
            break


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    from types import SimpleNamespace

    config = dict(
        seed=42, batch_size=32, num_workers=2, max_length=128,
        dataset="rotten_tomatoes", model="distilbert", device="cpu",
    )
    opts = SimpleNamespace(**config)
    set_seeds(opts.seed)

    with launch_ipdb_on_exception():
        main(opts)
