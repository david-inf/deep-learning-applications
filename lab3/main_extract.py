"""BERT as a feture extractor"""

import os
import numpy as np

from sklearn.svm import LinearSVC
from rich.console import Console
from rich.table import Table

from datasets import load_dataset
from transformers import set_seed
from models import get_distilbert_features, get_sbert_features


def get_dataset():
    """Dataset splits"""
    rt_trainset, rt_valset, rt_testset = load_dataset(
        "cornell-movie-review-data/rotten_tomatoes",
        split=["train", "validation", "test"])
    return rt_trainset, rt_valset, rt_testset


def extract_features(opts, texts, path):
    """Extract features and save locally for later use"""
    if opts.extractor == "distilbert":
        features = get_distilbert_features(opts, texts)
    elif opts.extractor == "sbert":
        features = get_sbert_features(opts, texts)
    else:
        raise ValueError(f"Unknown extractor {opts.extractor}")
    np.savetxt(path, features, delimiter=" ")


def main(opts):
    """Extract features and train classifier"""
    set_seed(opts.seed)

    path = "data/rt_features"
    features_path = os.path.join(path, opts.extractor)
    if opts.extract:
        os.makedirs(path, exist_ok=True)
        os.makedirs(features_path, exist_ok=True)
        # Get train-val-test splits
        rt_trainset, rt_valset, _ = get_dataset()

        train_labels = np.savetxt(os.path.join(
            path, "train_labels.txt"), np.array(rt_trainset["label"]))
        val_labels = np.savetxt(os.path.join(
            path, "val_labels.txt"), np.array(rt_valset["label"]))

        extract_features(opts, rt_trainset["text"], os.path.join(
            features_path, f"{opts.method}_train.txt"))
        extract_features(opts, rt_valset["text"], os.path.join(
            features_path, f"{opts.method}_val.txt"))

    train_features = np.loadtxt(os.path.join(
        features_path, f"{opts.method}_train.txt"))
    val_features = np.loadtxt(os.path.join(
        features_path, f"{opts.method}_val.txt"))

    train_labels = np.loadtxt(os.path.join(path, "train_labels.txt"))
    val_labels = np.loadtxt(os.path.join(path, "val_labels.txt"))

    # Train classifier and do inference
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    train_acc = clf.score(train_features, train_labels)
    val_acc = clf.score(val_features, val_labels)

    console = Console()
    table = Table(title="LogisticRegression classifier")
    table.add_column("Split")
    table.add_column("Accuracy")
    table.add_row("Train", str(train_acc))
    table.add_row("Val", str(val_acc))
    console.print(table)


def view_embeds(opts):
    """Visualize embeddings"""
    from sklearn.preprocessing import MinMaxScaler
    from umap import UMAP
    import matplotlib.pyplot as plt

    output_dir = "data/rt_features"
    train_embeds = np.loadtxt(os.path.join(
        output_dir, opts.extractor, f"{opts.method}_train.txt"))
    train_labels = np.loadtxt(os.path.join(output_dir, "train_labels.txt"))

    scaler = MinMaxScaler()
    scaled_train_embeds = scaler.fit_transform(train_embeds)

    reducer = UMAP(n_components=2, metric="cosine")
    embeddings = reducer.fit_transform(scaled_train_embeds)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    fig.suptitle(f"UMAP projection of {opts.extractor} ({opts.method}) embeddings")
    axs = axs.flatten()
    colormaps = ["Blues", "Greens"]
    for label, (ax, color) in enumerate(zip(axs.flatten(), colormaps)):
        subset = embeddings[train_labels == label]
        ax.hexbin(subset[:, 0], subset[:, 1], gridsize=20, cmap=color)
        ax.set_title(f"Label: {label}")
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    output_dir = os.path.join("lab3/results", opts.extractor + "_embeds.svg")
    plt.savefig(output_dir)


if __name__ == "__main__":
    from types import SimpleNamespace
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true",
                        help="Run when extracting from a new extractor,"
                        "this just to save the features locally")
    parser.add_argument("--extractor", default="distilbert",
                        choices=["distilbert", "sbert"], help="Feature extractor")
    parser.add_argument("--method", default="cls",
                        help="DistilBert: use CLS or mean pooling method. SBERT: choose model.",
                        choices=["cls", "mean", "mpnet", "minilm"])
    parser.add_argument("--view", action="store_true",
                        help="Visualize embeddings")
    args = parser.parse_args()

    configs = dict(seed=42, batch_size=32, device="cuda",
                   extract=args.extract, view=args.view,
                   extractor=args.extractor, method=args.method)
    args = SimpleNamespace(**configs)

    try:
        if not args.view:
            main(args)
        else:
            view_embeds(args)
    except Exception as e:
        import ipdb
        print(e)
        ipdb.post_mortem()
