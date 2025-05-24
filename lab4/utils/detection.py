"""Utilities for OOD detection"""

import torch
import torch.nn.functional as F


def compute_scores(opts, model, loader):
    """Forward passes given the model and score function"""
    model.eval()
    scores = []
    with torch.no_grad():
        for X, _ in loader:

            # move data to device
            X = X.to(opts.device)
            # forward pass
            output = model(X)

            # compute scores for this batch
            if opts.score_fun == "max_logit":
                # CNN (logits): [N, K]
                scores_b = max_logit(output)  # [N]
            elif opts.score_fun == "max_softmax":
                # CNN (logits): [N, K]
                scores_b = max_softmax(opts, output)
            elif opts.score_fun == "mse":
                # AE (images): [3, 28, 28]
                scores_b = mse(X, output)
            else:
                raise ValueError(f"Unknown score function {opts.score_fun}")

            scores.append(scores_b)

    return torch.cat(scores)


def max_logit(logits):
    """Max logit per sample for the CNN"""
    # get the max logit for each sample
    # the max function returns values and indices
    # so we take the values
    scores = logits.max(dim=1).values  # [N]
    return scores


def max_softmax(opts, logits):
    """Max softmax per sample for the CNN"""
    probs = F.softmax(logits / opts.temp, 1)
    scores = probs.max(dim=1)[0] # get the max for each element of the batch
    return scores


def mse(img, img_rec):
    """MSE for the AutoEncoder"""
    criterion = torch.nn.MSELoss(reduction='none')
    loss = criterion(img, img_rec)
    scores = loss.mean([1, 2, 3])
    return scores
