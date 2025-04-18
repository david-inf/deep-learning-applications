
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def get_bert(opts):
    if opts.model == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2)
        model = model.to(opts.device)
    else:
        raise ValueError(f"Unknown model {opts.model}")

    return tokenizer, model
