
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from utils import set_seeds, LOG


def get_dataset():
    """Dataset splits"""
    rt_trainset, rt_valset, rt_testset = load_dataset(
        "cornell-movie-review-data/rotten_tomatoes",
        split=["train", "validation", "test"])

    return rt_trainset, rt_valset, rt_testset


def distilbert_features(opts, texts):
    """Use DistilBERT as feature extractor"""
    checkpoint="distilbert-base-uncased"
    feature_extractor = pipeline(
        model=checkpoint, tokenizer=checkpoint, task="feature-extraction",
        framework="pt", device="cuda", batch_size=32,
        tokenize_kwargs=dict(max_length=128, truncation=True))
    extractions = feature_extractor(texts, return_tensors="pt")

    features = []
    for extract in extractions:  # get a tensor
        if opts.method == "cls":
            # extract CLS token
            features.append(extract[0].numpy()[0])
        elif opts.method == "mean":
            # mean pooling over each token embeddings
            features.append(extract[0].numpy().mean(axis=0))
        else:
            raise ValueError(f"Unknown extraction method {opts.method}")

    return np.vstack((features))


def sbert_features(opts, texts):
    """Use SentenceBERT as feature extractor"""
    if opts.method == "mpnet":
        model = SentenceTransformer("all-mpnet-base-v2", device=opts.device)  # [N, 768]
    elif opts.method == "minilm":
        model = SentenceTransformer("all-MiniLM-L6-v2", device=opts.device)  # [N, 384]
    else:
        raise ValueError(f"Unknown sbert {opts.method}")
    model.eval()
    features = model.encode(texts, show_progress_bar=True, batch_size=32)
    return np.vstack(features)


def main(opts):
    """Extract features and train classifier"""
    set_seeds(opts.seed)
    # Get train-val-test splits
    rt_trainset, rt_valset, rt_testset = get_dataset()

    # Extract features
    if opts.extractor == "distilbert":
        train_features = distilbert_features(opts, rt_trainset["text"])
        val_features = distilbert_features(opts, rt_valset["text"])
        test_features = distilbert_features(opts, rt_testset["text"])
    elif opts.extractor == "sbert":
        train_features = sbert_features(opts, rt_trainset["text"])
        val_features = sbert_features(opts, rt_valset["text"])
        test_features = sbert_features(opts, rt_testset["text"])
    else:
        raise ValueError(f"Unknown extractor {opts.extractor}")

    train_labels = np.array(rt_trainset["label"])
    val_labels = np.array(rt_valset["label"])
    test_labels = np.array(rt_testset["label"])

    # Train classifier and do inference
    if opts.classifier == "svm":
        LOG.info("LinearSVC")
        clf = LinearSVC()
    elif opts.classifier == "logistic":
        LOG.info("LogisticRegression")
        clf = LogisticRegression()
    else:
        raise ValueError(f"Unknown classifier {opts.classifier}")

    clf.fit(train_features, train_labels)
    train_acc = clf.score(train_features, train_labels)
    val_acc = clf.score(val_features, val_labels)
    test_acc = clf.score(test_features, test_labels)

    LOG.info("train_acc=%.3f", train_acc)
    LOG.info("val_acc=%.3f", val_acc)
    LOG.info("test_acc=%.3f", test_acc)


if __name__ == "__main__":
    from types import SimpleNamespace
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", default="distilbert", choices=["distilbert", "sbert"], help="Feature extractor")
    parser.add_argument("--method", default="cls",
                        help="DistilBert: use CLS or mean pooling method. SBERT: choose model.",
                        choices=["cls", "mean", "mpnet", "minilm"])
    parser.add_argument("--classifier", default="svm", choices=["svm", "logistic"],
                        help="Classifier to build ontop of the feature extractor")
    args = parser.parse_args()

    configs = dict(seed=42, batch_size=32, device="cuda",
                   extractor=args.extractor, method=args.method,
                   classifier=args.classifier)
    args = SimpleNamespace(**configs)

    try:
        main(args)
    except Exception:
        import ipdb
        ipdb.post_mortem()
