
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def get_bert(opts):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)
    model = model.to(opts.device)

    if opts.ft_setting == "full":
        # train all parameters
        pass
    elif opts.ft_setting == "head":
        # i.e. logistic regression
        # freeze embeddings and transformer layers
        # leave only pre-classifier and classifier trainable
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for param in model.distilbert.parameters():
            param.requires_grad = False
    elif opts.ft_setting == "custom":
        # freeze the first 4 out of 6 transformer layers
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for i in range(4):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Unknown training setting {opts.ft_setting}")

    return tokenizer, model
