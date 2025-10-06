"""Module implementing loss based on Cosine similarity."""

import torch

from torch.nn import CosineSimilarity


class LossWrapper(torch.nn.Module):
    """Wrapper around a regular loss to add a mask input.

    Parameters
    ----------
    loss_fn : nn.Module
        loss, needs to be unreduced.
    """

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, preds, labels, mask):
        """Forward function."""
        loss = self.loss(preds, labels)
        return self.loss(preds, labels)


class CosineSimilarityLoss(torch.nn.Module):
    """Loss based on cosine similarity used to train models.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.loss_fn = CosineSimilarity(dim=1, eps=eps)

    def forward(self, pred, label, mask=None):
        """Compute forward pass."""
        loss = 1 - self.loss_fn(pred, label)
        loss = torch.mean(loss)
        return loss
