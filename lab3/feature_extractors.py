
import torch
import numpy as np
from tqdm import tqdm

from transformers import DistilBertTokenizer, DistilBertModel, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from utils import N
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# TODO: feature extraction from huggingface?

def _get_bert(opts):
    """DistilBERT"""
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model = model.to(opts.device)
    return model, tokenizer


def bert_extractor(opts, dataset):
    model, tokenizer = _get_bert(opts)
    model.eval()

    # tokenize dataset
    def preprocess(sample):
        return tokenizer(
            # tokenize the text without padding
            sample["text"],
            # truncate to specified length if necessary
            max_length=128,
            truncation=True,
            return_attention_mask=True,
            # returns lists as the default collator wants
            return_tensors=None,
        )
    tokenized_dataset = dataset.map(
        preprocess, batched=True, num_proc=2,
        remove_columns=["text"], desc="Tokenizing")
    collate_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            # dynamic padding, different per each batch
            padding="longest")
    loader = DataLoader(tokenized_dataset, batch_size=opts.batch_size,
                        num_workers=2, pin_memory=True, collate_fn=collate_fn)

    features_list = []
    with tqdm(loader, unit="batch", desc="BERT") as tepoch:
        for batch in tepoch:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(opts.device)
                attention_mask = batch["attention_mask"].to(opts.device)
                y = batch["labels"].to(opts.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # TODO: try two options (i) CLS (ii) pooling
                # take the [CLS] token -> [1, 768]
                features = outputs.last_hidden_state[:, 0, :]
                # mean pooling -> [1, 768]
                # features = outputs.last_hidden_state.mean(dim=1)
                features_list.append(N(features))

    return np.vstack(features_list)  # [N, 768]


def _get_sbert(opts):
    """SentenceBERT"""
    # model = SentenceTransformer("all-mpnet-base-v2", device=opts.device)  # [N, 768]
    model = SentenceTransformer("all-MiniLM-L6-v2", device=opts.device)  # [N, 384]
    return model


def sbert_extractor(opts, texts):
    model = _get_sbert(opts)
    model.eval()

    features_list = []
    for i in tqdm(range(0, len(texts), opts.batch_size), desc="SBERT"):
        batch = texts[i:i+opts.batch_size]
        # no tokenizer required
        # outputs the embeddings as defined
        features = model.encode(batch, show_progress_bar=False)
        features_list.append(features)

    return np.vstack(features_list)
