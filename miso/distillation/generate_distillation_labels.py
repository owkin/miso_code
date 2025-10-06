import torch
from torch.utils.data import DataLoader, Subset

from pathlib import Path

import numpy as np

import pandas as pd

from miso.data.loading import DatasetSlide
from miso.models import LocalAttentionMIL

from tqdm import tqdm

import argparse

import hydra

from omegaconf import DictConfig, OmegaConf


def generate_pseudolabels(cfg):

    cfg = OmegaConf.to_container(cfg)
    dataset = hydra.utils.instantiate(cfg["dataset"])
    cfg["model"]["input_dim"] = dataset.in_dim
    cfg["model"]["output_dim"] = dataset.out_dim

    n_repeats = cfg["training"]["n_repeats"]
    n_folds = cfg["training"]["n_folds"]

    models = []
    for r in range(n_repeats):
        for f in range(n_folds):
            model = model = hydra.utils.instantiate(cfg["model"])

            model.load_state_dict(
                torch.load(
                    Path(cfg["training"]["save_dir"]) / f'repeat_{r}_fold_{f}/model.pth',
                    map_location=cfg["model"]["device"]
                )
            )
            models.append(model)

    preds = {}
    for name in dataset.slide_names:
        preds[name] = []

    df_results = pd.read_csv(
        Path(cfg["training"]["save_dir"]) / 'metrics.csv'
    )

    for k in range(n_repeats * n_folds):
        
        model = models[k]
        r = k // n_folds
        f = k % n_folds

        train_ids = [
            name for name in dataset.slide_names
            if f'correlation-pearson_{name}_fold_{f}_repeat_{r}' not in df_results.columns
        ]
        idx = np.arange(len(dataset))[pd.Series(dataset.slide_names).isin(train_ids).values]
        train_set = Subset(dataset, idx)
        loader = DataLoader(train_set, batch_size=1, num_workers=8)

        with torch.no_grad():
            for data in tqdm(loader):
                x = data['features'].float().to(cfg["model"]["device"])
                n = data['neighbors'].long().to(cfg["model"]["device"])
                c = data['coords'].long()
                y = data['labels'].float()
                name = data['slide_id'][0]
                mask = (torch.sum(x**2, keepdim=True, dim=2) == 0)
                pred = model(x, n, mask)
                pred = pred[0, ~mask[0, :, 0]].cpu().numpy()
                preds[name].append(pred)

    preds_aggregated = {}
    items = list(preds.items())
    for key, value in items:
        preds_aggregated[key] = np.mean(value, axis=0)
        del preds[key]

    path = Path(cfg["dataset"]["path_to_counts"])
    path.mkdir(exist_ok=True)
    for key, value in preds_aggregated.items():
        (path / key).mkdir(exist_ok=True)
        (path / key / 'pseudolabels').mkdir(exist_ok=True)
        barcode = dataset.barcodes[dataset.slide_names_list == key]
        np.save(path / key / 'pseudolabels' / 'reads.npy', value)
        np.save(path / key / 'pseudolabels' / 'barcodes.npy', barcode)
        np.save(path / key / 'pseudolabels' / 'gene_names.npy', dataset.gene_names)

@hydra.main(
    config_path="../confs",
    config_name="train",
    version_base=None,
)        
def main(
    cfg: DictConfig
):
    generate_pseudolabels(cfg)
        
if __name__ == '__main__':
    main()
