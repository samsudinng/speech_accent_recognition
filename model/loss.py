import torch.nn.functional as F
import torch
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def ce_loss(output, target):
    return F.cross_entropy(output, target)


"""
LABEL SMOOTHING CROSS-ENTROPY LOSS

Adapted from:
    https://github.com/wangleiofficial/label-smoothing-pytorch
"""

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def ce_labelsmoothing_loss(preds, target, epsilon: float = 0.1, reduction='mean'):
    n = preds.size()[-1]
    log_preds = F.log_softmax(preds, dim=-1)
    loss = reduce_loss(-log_preds.sum(dim=-1), reduction)
    nll = F.nll_loss(log_preds, target, reduction=reduction)
    return linear_combination(loss / n, nll, epsilon)

