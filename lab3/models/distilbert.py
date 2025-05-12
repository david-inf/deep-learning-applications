
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import LoraConfig, TaskType, get_peft_model
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


def get_distilbert(opts):
    """
    Get BERT pretrained model and its tokenizer for finetuning
    - DistilBERT
    - SciBERT
    """
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2)
    
    tokenizer, model = _finetuning_setting(opts, tokenizer, model)
    return tokenizer, model


def _finetuning_setting(opts, tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification):
    """Set BERT model finetuning settings"""
    ft_setting = SimpleNamespace(**opts.ft_setting)

    if ft_setting.type == "full":
        # train all parameters
        pass

    elif ft_setting.type == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=ft_setting.rank, lora_alpha=ft_setting.alpha,
            lora_dropout=0.0, bias="none",
            target_modules=ft_setting.target_modules
        )
        model = get_peft_model(model, peft_config)

    else:
        raise ValueError(f"Unknown training setting {opts.ft_setting}")

    return tokenizer, model
