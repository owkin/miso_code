"""Metrics for evaluating the performance of a model."""

import warnings

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings("ignore")


def stats_pearson_corr(pred, label, i):
    """Enable use of delayed."""
    return stats.pearsonr(label[:, i], pred[:, i])


def stats_spearman_corr(pred, label, i):
    """Enable use of delayed."""
    return stats.spearmanr(label[:, i], pred[:, i])


def parallel_metrics(label, pred, metric):
    """Compute a given metric.

    Enables parallel computation of a given metric comparing labels and preds.

    Parameters
    ----------
    label : _type_
        _description_
    pred : _type_
        _description_
    metric : _type_, optional
        _description_, by default spearman_corr

    Returns
    -------
    np.array
        _description_
    """
    res = Parallel(n_jobs=8)(
        delayed(metric)(pred, label, i) for i in range(label.shape[1])
    )
    res_ = np.array(res)
    res_[np.isnan(res_)] = 0
    return res_


def compute_metrics(preds, labels, slide_ids, masks, fold, repeat, gene_list):
    report = {"gene": gene_list}

    for slide_id in np.unique(slide_ids):
        pred = preds[slide_ids == slide_id].squeeze()
        label = labels[slide_ids == slide_id].squeeze()
        mask = masks[slide_ids == slide_id].squeeze()
        # When training super-resolution models mask is not relevant
        if mask.ndim == 2:
            mask = np.zeros(pred.shape[0], dtype='bool')

        temp_p = parallel_metrics(label[~mask], pred[~mask], metric=stats_pearson_corr)
        temp_s = parallel_metrics(label[~mask], pred[~mask], metric=stats_spearman_corr)

        mae = torch.nn.L1Loss(reduction="none")(
            torch.from_numpy(pred), torch.from_numpy(label)
        )
        mse = torch.nn.MSELoss(reduction="none")(
            torch.from_numpy(pred), torch.from_numpy(label)
        )

        report[f"correlation-pearson_{slide_id}_fold_{fold}_repeat_{repeat}"] = temp_p[
            :, 0
        ]
        report[f"p-val-pearson_{slide_id}_fold_{fold}_repeat_{repeat}"] = temp_p[:, 1]

        report[f"correlation-spearman_{slide_id}_fold_{fold}_repeat_{repeat}"] = temp_s[
            :, 0
        ]
        report[f"p-val-spearman_{slide_id}_fold_{fold}_repeat_{repeat}"] = temp_s[:, 1]

        report[f"mae_{slide_id}_fold_{fold}_repeat_{repeat}"] = (
            mae.sum(0) / mae.shape[0]
        )
        report[f"rmse_{slide_id}_fold_{fold}_repeat_{repeat}"] = torch.sqrt(
            mse.sum(0) / mse.shape[0]
        )

    return pd.DataFrame(report).set_index("gene")
