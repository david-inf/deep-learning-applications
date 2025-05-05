
from types import SimpleNamespace
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model


def get_bert(opts):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)
    ft_setting = SimpleNamespace(**opts.ft_setting)

    if ft_setting.type == "full":
        # train all parameters
        pass

    elif ft_setting.type == "head":
        # i.e. 2-layers MLP
        # freeze embeddings and transformer layers
        # leave only pre-classifier and classifier trainable
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for param in model.distilbert.parameters():
            param.requires_grad = False

    elif ft_setting.type == "custom":
        # freeze the first 4 out of 6 transformer layers
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        for i in range(4):
            for param in model.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

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

    model = model.to(opts.device)
    return tokenizer, model
