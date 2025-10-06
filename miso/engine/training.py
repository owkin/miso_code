"""Fit function for training a model."""

from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm
import mlflow


def fit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset | None,
    params: dict[str, any],
):
    n_epochs = params["n_epochs"]
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    for epoch in range(n_epochs):
        train_loss = training_epoch(model, optimizer, criterion, train_loader)
        mlflow.log_metric("train_loss", train_loss, step=epoch)

    preds, labels, slide_ids, masks = predict(model, test_loader)

    return preds, labels, slide_ids, masks


def training_epoch(model, optimizer, loss_fn, dataloader):
    """Train model for one epoch."""
    model.train()
    train_loss = []
    for batch in tqdm(dataloader, desc="Training..."):
        feats, labels = batch["features"], batch["labels"]
        feats = feats.float().to(model.device)
        labels = labels.float().to(model.device)
        # Mask padded values
        mask = torch.sum(feats**2, dim=-1, keepdim=True) == 0.0
        if "neighbors" in batch.keys():
            neighbors = batch["neighbors"].long().to(model.device)
            pred = model(feats, neighbors, mask)
        else:
            pred = model(feats, mask)

        loss = loss_fn(pred, labels, mask)

        train_loss += [loss.detach().cpu().numpy()]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_train_loss = np.mean(train_loss)
    return mean_train_loss


def predict(model, dataloader):
    """Perform prediction on the test set."""
    model.eval()
    labels_list, preds, coords_list, mask_list, slide_ids_list = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            feats, labels = batch["features"], batch["labels"]
            slide_ids = batch["slide_id"]
            labels = labels.float()
            feats = feats.float().to(model.device)
            # Mask padded values
            mask = torch.sum(feats**2, dim=-1, keepdim=True) == 0.0
            if "neighbors" in batch.keys():
                neighbors = batch["neighbors"].long().to(model.device)
                pred = model(feats, neighbors, mask)
            else:
                pred = model(feats, mask)
            pred = torch.nn.ReLU()(pred)

            # Convert to numpy
            pred = pred.cpu().numpy()

            labels_list += [labels.detach().cpu().numpy()]
            slide_ids_list += list(slide_ids)
            preds += [pred]
            mask_list += [mask.cpu().numpy()]

    slide_ids = np.array(slide_ids_list)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels_list)
    masks = np.concatenate(mask_list)

    return preds, labels, slide_ids, masks
