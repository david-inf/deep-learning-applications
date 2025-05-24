"""OOD detection and performance evaluation"""

# Update imports to use relative or absolute paths
import sys
import os
from types import SimpleNamespace

import yaml
import torch
from torch.nn import Module
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path

from lab4.mydata import get_loaders
from lab4.utils.detection import compute_scores
from lab4.models import AutoEncoder

from lab1.main_train import get_model
from lab1.utils.train import load_checkpoint
from lab1.utils import set_seeds, LOG


def score_distrib(opts, model: Module, id_loader, ood_loader, path):
    """Plot the scores distribution for ID and OOD samples"""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"Scores using {opts.score_fun}")

    id_lab = "ID samples"
    scores_id = compute_scores(opts, model, id_loader)
    ood_lab = "OOD samples"
    scores_ood = compute_scores(opts, model, ood_loader)

    axs[0].plot(sorted(scores_id.cpu()), label=id_lab)
    axs[0].plot(sorted(scores_ood.cpu()), label=ood_lab)
    axs[0].set_xlabel("Ordered samples")
    axs[0].set_ylabel("Scores")
    axs[0].legend()

    axs[1].hist(scores_id.cpu(), density=True,
                alpha=0.5, bins=25, label=id_lab)
    axs[1].hist(scores_ood.cpu(), density=True,
                alpha=0.5, bins=25, label=ood_lab)
    axs[1].set_xlabel("Scores")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(path)
    LOG.info("Scores distribution at path=%s", {path})

    return scores_id, scores_ood


def evaluate(scores_id, scores_ood):
    """Plot ROC and PR curves"""
    pred = torch.cat((scores_id, scores_ood))
    gt = torch.cat((torch.ones_like(scores_id), torch.zeros_like(scores_ood)))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # ROC curve
    RocCurveDisplay.from_predictions(gt.numpy(), pred.numpy(), ax=axs[0])
    axs[0].set_title("ROC Curve")

    # PR curve
    PrecisionRecallDisplay.from_predictions(gt.numpy(), pred.numpy(), ax=axs[1])
    axs[1].set_title("Precision-Recall Curve")
    
    plt.tight_layout()
    
    path = "lab4/plots/scores_roc_pr.svg"
    plt.savefig(path)
    LOG.info("ROC and PR curves at path=%s", {path})


def main(opts):
    """OOD detection pipeline"""
    set_seeds(opts.seed)
    # Load model checkpoint
    with open(opts.model_configs, "r", encoding="utf-8") as f:
        model_configs = yaml.safe_load(f)
    model_opts = SimpleNamespace(**model_configs)
    model_opts.device = opts.device

    # Load model
    if model_opts.model == "CNN":
        model = get_model(model_opts)
    elif model_opts.model == "AutoEncoder":
        model = AutoEncoder(model_opts.num_filters)
        model = model.to(model_opts.device)
    else:
        raise ValueError(f"Unknown model {model_opts.model}")
    load_checkpoint(model_opts.checkpoint, model, opts.device)

    # Load data
    id_loader, ood_loader = get_loaders(model_opts)

    # Distribution on ID and OOD samples
    output_dir = "lab4/plots"
    path = os.path.join(output_dir, f"scores_{opts.score_fun}_{model_opts.model}.svg")
    out = score_distrib(opts, model, id_loader, ood_loader, path)

    # Performance evaluation
    evaluate(out[0], out[1])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OOD detection pipeline")
    parser.add_argument("--score_fun", type=str, default="max_logit",
                        help="Score function to use (default: max_logit)",
                        choices=["max_logit", "max_softmax", "mse"])
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Temperature for softmax (default: 1.0)")
    parser.add_argument("--model_configs", type=str,
                        help="Model configuration file")
    args = parser.parse_args()
    args.seed = 42
    try:
        main(args)
    except Exception:
        import ipdb
        import traceback
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
