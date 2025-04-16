
import numpy as np
import torch
from sklearn.svm import LinearSVC
from datasets import load_dataset
from feature_extractors import bert_extractor, sbert_extractor
from utils import set_seeds, LOG


def get_dataset():
    rt_trainset, rt_valset, rt_testset = load_dataset(
        "cornell-movie-review-data/rotten_tomatoes",
        split=["train", "validation", "test"])

    return rt_trainset, rt_valset, rt_testset


def main(opts):
    set_seeds(opts.seed)
    # Get train-val-test sets
    rt_trainset, rt_valset, rt_testset = get_dataset()

    # Extract features
    if opts.model == "bert":
        train_features = bert_extractor(opts, rt_trainset["text"])
        val_features = bert_extractor(opts, rt_valset["text"])
        test_features = bert_extractor(opts, rt_testset["text"])
    elif opts.model == "sbert":
        train_features = sbert_extractor(opts, rt_trainset["text"])
        val_features = sbert_extractor(opts, rt_valset["text"])
        test_features = sbert_extractor(opts, rt_testset["text"])
    else:
        raise ValueError(f"Unknown extractor {opts.model}")

    train_labels = np.array(rt_trainset["label"])
    val_labels = np.array(rt_valset["label"])
    test_labels = np.array(rt_testset["label"])

    # Train classifier
    svm = LinearSVC()
    svm.fit(train_features, train_labels)

    # Inference
    LOG.info(f"{opts.model.upper()} train_acc={svm.score(train_features, train_labels):.3f}")
    LOG.info(f"{opts.model.upper()} val_acc={svm.score(val_features, val_labels):.3f}")
    LOG.info(f"{opts.model.upper()} test_acc={svm.score(test_features, test_labels):.3f}")
    # TODO: being of the same dimensionality, what about a distance/similary 
    # measure between the two embeddings?


if __name__ == "__main__":
    from types import SimpleNamespace
    import argparse
    from ipdb import launch_ipdb_on_exception

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Feature extractor")
    args = parser.parse_args()

    config = dict(seed=42, device="cuda")
    opts = SimpleNamespace(**config)
    opts.model = args.model

    with launch_ipdb_on_exception():
        main(opts)
