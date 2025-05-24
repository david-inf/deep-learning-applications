"""Utilities for OOD detection"""

import torch
import torch.nn.functional as F


def max_logit(opts, logit):
    """Max logit per sample"""
    # get the max logit for each sample
    # the max function returns values and indices
    # so we take the values
    scores = logit.max(dim=1).values  # [N]
    return scores


def max_softmax(opts, logit):
    """Max softmax per sample"""
    probs = F.softmax(logit / opts.temp, 1)
    scores = probs.max(dim=1)[0] # get the max for each element of the batch
    return scores


def compute_scores(opts, model, loader):
    """Forward passes given the model and score function"""
    # choose the function to compute scores
    if opts.score_fun == "max_logit":
        score_fun = max_logit
    elif opts.score_fun == "max_softmax":
        score_fun = max_softmax
    else:
        raise ValueError(f"Unknown score function {opts.score_fun}")

    # compute scores
    scores = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(opts.device)
            output = model(X)  # logits [N, K]
            # compute scores for this batch
            scores_b = score_fun(opts, output)  # [N]
            scores.append(scores_b)

    return torch.cat(scores)
