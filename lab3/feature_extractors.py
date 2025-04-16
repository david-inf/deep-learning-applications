
import torch
import numpy as np
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from utils import N
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# TODO: feature extraction from huggingface?

def _get_bert(opts):
    # DistilBERT
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model = model.to(opts.device)
    return model, tokenizer


def bert_extractor(opts, texts):
    model, tokenizer = _get_bert(opts)
    model.eval()
    features_list = []

    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT"):
        batch = texts[i:i+batch_size]
        encoded_batch = tokenizer(
            batch,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(opts.device)

        with torch.no_grad():
            outputs = model(**encoded_batch)
            # take the [CLS] token
            features = outputs.last_hidden_state[:, 0, :]
            features_list.append(N(features))

    return np.vstack(features_list)


def _get_sbert(opts):
    # SBERT
    model = SentenceTransformer("all-mpnet-base-v2", device=opts.device)
    return model


def sbert_extractor(opts, texts):
    model = _get_sbert(opts)
    model.eval()
    features_list = []

    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="SBERT"):
        batch = texts[i:i+batch_size]
        # no tokenizer required
        # outputs the embeddings as defined
        features = model.encode(batch, show_progress_bar=False)
        features_list.append(features)

    return np.vstack(features_list)
