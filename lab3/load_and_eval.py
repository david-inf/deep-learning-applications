"""Load model from checkpoint and evaluate on testset"""

from main_ft import get_loaders, get_model
from utils import LOG
from utils.train_utils import load_model
from train import test


def main(opts):
    """Load model from checkpoint, then evaluate on testset"""
    # Load finetuned model
    tokenizer, model = get_model(opts)
    # TODO: load checkpoint
    # Load test set
    _, _, test_loader = get_loaders(opts)

    # Evaluate
    LOG.info("val_acc=")


if __name__ == "__main__":
    import cmd_args
    opts = cmd_args.parse_args()
    opts.device = "cuda"
    try:
        main(opts)
    except Exception as e:
        print(e)
