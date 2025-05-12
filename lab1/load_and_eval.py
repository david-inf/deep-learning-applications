"""Load model from checkpoint and evaluate on valset"""

from main_train import get_loaders, get_model
from utils import LOG
from utils.train_utils import load_checkpoint
from train import test


def main(opts):
    """Load model and evaluate quickly"""
    # Validation set and model to evaluate
    _, val_loader = get_loaders(opts)
    model = get_model(opts)
    load_checkpoint(opts.checkpoint, model)

    # Evaluate
    _, val_acc = test(opts, model, val_loader)
    LOG.info("val_acc=%.3f", val_acc)


if __name__ == "__main__":
    import cmd_args
    opts = cmd_args.parse_args()
    opts.device = "cuda"
    try:
        main(opts)
    except Exception as e:
        print(e)
