
import numpy as np
import torch

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# from models.mlp import MLP2

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
    if opts.extractor == "bert":
        train_features = bert_extractor(opts, rt_trainset)
        val_features = bert_extractor(opts, rt_valset)
        test_features = bert_extractor(opts, rt_testset)
    elif opts.extractor == "sbert":
        train_features = sbert_extractor(opts, rt_trainset["text"])
        val_features = sbert_extractor(opts, rt_valset["text"])
        test_features = sbert_extractor(opts, rt_testset["text"])
    else:
        raise ValueError(f"Unknown extractor {opts.extractor}")

    train_labels = np.array(rt_trainset["label"])
    val_labels = np.array(rt_valset["label"])
    test_labels = np.array(rt_testset["label"])

    # Train classifier and do inference
    if opts.model == "svm":
        clf = LinearSVC()
        clf.fit(train_features, train_labels)
        LOG.info("LinearSVC")
        train_acc = clf.score(train_features, train_labels)
        val_acc = clf.score(val_features, val_labels)
        test_acc = clf.score(test_features, test_labels)
    elif opts.model == "logistic":
        clf = LogisticRegression()
        clf.fit(train_features, train_labels)
        LOG.info("LogisticRegression")
        train_acc = clf.score(train_features, train_labels)
        val_acc = clf.score(val_features, val_labels)
        test_acc = clf.score(test_features, test_labels)
    # elif opts.model == "mlp2":
    #     model = MLP2(train_features.shape[1])
    #     model = model.to("cuda")
    else:
        raise ValueError(f"Unknown model {opts.model}")


    LOG.info(f"{opts.extractor.upper()} train_acc={train_acc:.3f}")
    LOG.info(f"{opts.extractor.upper()} val_acc={val_acc:.3f}")
    LOG.info(f"{opts.extractor.upper()} test_acc={test_acc:.3f}")


if __name__ == "__main__":
    from types import SimpleNamespace
    import argparse
    from ipdb import launch_ipdb_on_exception

    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", help="Feature extractor")
    parser.add_argument("--model", help="Model to build ontop of the feature extractor")
    args = parser.parse_args()

    config = dict(seed=42, batch_size=32, device="cuda",
                  extractor=args.extractor, model=args.model)
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
