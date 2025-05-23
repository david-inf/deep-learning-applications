"""Load model from checkpoint and evaluate on testset"""

from comet_ml import start
from accelerate import Accelerator
from torch.backends import cudnn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from main_ft import get_loaders
from utils import LOG
from train import test


def get_model(opts):
    """Load model from checkpoint"""
    tokenizer = AutoTokenizer.from_pretrained(opts.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        opts.checkpoint, num_labels=2)
    return tokenizer, model


def main(opts):
    """Load model from checkpoint, then evaluate on testset"""
    # Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    LOG.info("Accelerator device=%s", accelerator.device)

    # Load finetuned model
    tokenizer, model = get_model(opts)
    # Load test set
    with accelerator.main_process_first():
        _, _, test_loader = get_loaders(opts, tokenizer)

    # Prepare for evaluation
    cudnn.benchmark = True
    model, test_loader = accelerator.prepare(
        model, test_loader)

    # Evaluate
    _, val_acc = test(model, accelerator, test_loader)
    LOG.info("val_acc=%.3f", val_acc)

    # Log to comet_ml
    experiment = start(project="deep-learning-applications")
    experiment.set_name(opts.experiment_name)
    experiment.log_parameters(vars(opts))
    experiment.log_metrics({
        "val_acc": val_acc,
    })
    experiment.end()


if __name__ == "__main__":
    import cmd_args
    opts = cmd_args.parse_args()
    opts.device = "cuda"
    try:
        main(opts)
    except Exception as e:
        print(e)
