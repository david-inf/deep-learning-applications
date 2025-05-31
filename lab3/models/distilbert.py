
from transformers import pipeline
import numpy as np


def get_distilbert_features(opts, texts):
    """Extract DistilBERT features for given texts"""
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
