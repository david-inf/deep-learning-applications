"""OOD detection and performance evaluation"""

from types import SimpleNamespace
import yaml

from lab1.utils.misc_utils import set_seeds
from lab1.main_train import get_model
from lab1.utils.train_utils import load_checkpoint

from mydata import get_loaders


def main(opts):
    set_seeds(opts.seed)
    # Load model checkpoint
    with open(opts.model_configs, "r", encoding="utf-8") as f:
        model_configs = yaml.safe_load(f)
    model_opts = SimpleNamespace(**model_configs)
    model = get_model(model_opts)
    load_checkpoint(opts.ckpt, model)

    # Load data
    id_loader, ood_loader = get_loaders(opts)

    # Evaluate on testset (ID)


    # Evaluate on fakeset (OOD)



if __name__ == "__main__":
    configs = {
        "seed": 42,
        "ckpt": "lab1/ckpts/CNN/LargeCNNskip.pt",
        "model_configs": "lab1/configs/CNN/LargeCNNskip.yaml",
    }
    opts = SimpleNamespace(**configs)
    try:
        main(opts)
    except Exception:
        import ipdb
        import traceback
        import sys
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])

