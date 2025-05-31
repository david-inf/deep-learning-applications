
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

from models.distilbert import get_distilbert_features
from models.sbert import get_sbert_features


def get_bert(opts):
    """
    Get BERT pretrained model and its tokenizer for finetuning
    - DistilBERT (uncased)
    - SentenceBERT (mpnet)
    """
    if opts.model == "distilbert":
        checkpoint = "distilbert-base-uncased"
    elif opts.model == "sbert":
        checkpoint = "sentence-transformers/all-mpnet-base-v2"
    else:
        raise ValueError(f"Uknown BERT-family model {opts.model}")

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
