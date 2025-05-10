
from sentence_transformers import SentenceTransformer
import numpy as np


def get_sbert_features(opts, texts):
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
