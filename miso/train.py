"""
Copyright (c) Owkin Inc.
This source code is licensed under the CC BY-NC-SA 4.0 license found in the
LICENSE file in the root directory of this source tree.

Script for training the model.
"""
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import mlflow
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from miso.engine import fit
from miso.metrics import compute_metrics
import pandas as pd
import numpy as np
import pickle as pkl


def train(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)

    # Setup mlflow
    exp_name = cfg["training"]["exp_name"]
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp:
        exp_id = exp.experiment_id
    else:
        print(f"Experiment with name {exp_name} not found. Creating it.")
        exp_id = mlflow.create_experiment(name=exp_name)

    mlflow_main_run = mlflow.start_run(experiment_id=exp_id, run_name="average")
    run_id = mlflow_main_run.info.run_id

    # Load the dataset
    dataset = hydra.utils.instantiate(cfg["dataset"])
    cfg["model"]["input_dim"] = dataset.in_dim
    cfg["model"]["output_dim"] = dataset.out_dim

    mlflow.log_params(cfg)
    mlflow.log_params({"run_id": run_id})

    n_repeats = cfg["training"]["n_repeats"]
    n_folds = cfg["training"]["n_folds"]
    random_state = cfg["training"]["random_state"]
    save_dir = Path(cfg["training"]["save_dir"])
    save_model = cfg["training"]["save_model"]
    if "splits" in cfg["training"].keys():
        splits = cfg["training"]["splits"]
    else:
        splits = None

    list_reports = []
    for repeat in range(n_repeats):
        
        if splits is None:
            kf = KFold(n_splits=n_folds, random_state=random_state + repeat, shuffle=True)
            splits = list(kf.split(dataset))
            splits = [[split[0], split[1], split[1]] for split in splits]
        else:
            split_ids = pkl.load(open(splits, 'rb'))
            splits = []
            for train_ids, val_ids, test_ids in split_ids:
                splits.append([
                    np.arange(len(dataset))[pd.Series(dataset.slide_names).isin(train_ids).values],
                    np.arange(len(dataset))[pd.Series(dataset.slide_names).isin(val_ids).values],
                    np.arange(len(dataset))[pd.Series(dataset.slide_names).isin(test_ids).values]
                ])

        for fold, (train_index, val_index, test_index) in enumerate(splits):
            logger.info(f"Repeat {repeat + 1}, Fold {fold + 1}")

            mlflow_fold_run = mlflow.start_run(nested=True, experiment_id=exp_id)
            mlflow.log_params(
                {
                    "parent_run_id": run_id,
                    "run_id": mlflow_fold_run.info.run_id,
                    "run_type": "fold",
                    "repetition": repeat,
                    "fold": fold,
                }
            )

            split_save_dir = save_dir / f"repeat_{repeat}_fold_{fold}"
            split_save_dir.mkdir(exist_ok=True, parents=True)

            train_set = Subset(dataset, train_index)
            test_set = Subset(dataset, test_index)

            # Load the model
            model = hydra.utils.instantiate(cfg["model"])

            # Load the optimizer
            params = list(model.parameters())
            trainable_params = [var_p for var_p in params if var_p.requires_grad]
            optimizer = hydra.utils.instantiate(cfg["optimizer"], trainable_params)

            # Load the loss function
            criterion = hydra.utils.instantiate(cfg["criterion"])

            # Train the model
            preds, labels, slide_ids, masks = fit(
                model,
                optimizer,
                criterion,
                train_set,
                test_set=test_set,
                params=cfg["fit_params"],
            )

            df_metrics = compute_metrics(
                preds, labels, slide_ids, masks, fold, repeat, dataset.gene_names
            )
            list_reports.append(df_metrics)

            df_metrics.to_csv(
                split_save_dir / f"metrics_repeat_{repeat}_fold_{fold}.csv"
            )

            if save_model:
                model_path = split_save_dir / "model.pth"
                torch.save(model.state_dict(), model_path)

            mlflow.end_run()

    df_reports = pd.concat(list_reports, axis=1)
    df_reports.to_csv(save_dir / "metrics.csv")

    logger.info("Training finished.")

    mlflow.end_run()


@hydra.main(
    config_path="confs",
    config_name="train",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
