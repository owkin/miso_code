"""Script to extract features from a Visium sample and its associated slide."""

from __future__ import annotations

from pathlib import Path
import argparse
from collections.abc import Callable
from loguru import logger
from tqdm import tqdm
import openslide
import pandas as pd
import numpy as np
import torch
import scanpy as sc
import os

from miso.data.processing.extract_features import main as extract_features
from miso.data.processing.compute_neighbors import main as compute_neighbors


def main(
    path_visium: str,
    path_slide: str,
    path_output_folder: str,
    level: int,
    knn: int,
    tile_size: int = 224,
    use_gpu: bool = False,
):
    """Main function to process data from a Visium sample and its associated slide.

    Parameters
    ----------
    path_visium : str
        path to the visium sample's directory.
    path_slide : Path
        path to the associated slide.
    path_output_folder : Path
        path where the new slide will be saved.
    level : int
        level of the slide to extract features from.
    knn : int
        number of neighbors to extract.
    tile_size : int
        size of the tiles to extract, default is 224.
    processor : Callable | None
        image processor to use, default is None. If None, the default processor is
        loaded.
    model : Callable | None
        model to use, default is None. If None, the default model is loaded.
    use_gpu : bool
        whether to use gpu, default is False.
    """
    path_visium = Path(path_visium)
    path_slide = Path(path_slide)
    path_output_folder = Path(path_output_folder)

    extract_features(
        path_visium=path_visium,
        path_slide=path_slide,
        path_output_folder=path_output_folder,
        level=level,
        tile_size=tile_size,
        use_gpu=use_gpu,
    )

    compute_neighbors(
        path_feats=path_output_folder / "features.npy",
        path_output_folder=path_output_folder,
        knn=knn,
    )

    path_h5 = list(path_visium.glob("*filtered_feature_bc_matrix.h5"))[0]

    reads_df = sc.read_10x_h5(path_h5).to_df()

    reads = reads_df.values
    genes = reads_df.columns.values
    barcodes = reads_df.index.values

    (path_output_folder / "raw_reads").mkdir(exist_ok=True)
    np.save(path_output_folder / "raw_reads" / "gene_names.npy", genes.astype(str))
    np.save(path_output_folder / "raw_reads" / "barcodes.npy", barcodes.astype(str))
    np.save(path_output_folder / "raw_reads" / "reads.npy", reads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_visium", type=str, required=True, help="path to visium sample."
    )
    parser.add_argument(
        "--path_slide", type=str, required=True, help="path to the associated slide."
    )
    parser.add_argument(
        "--path_output_folder",
        type=str,
        required=True,
        help="path where the new slide will be saved.",
    )
    parser.add_argument(
        "--level",
        type=int,
        required=True,
        help="level of the slide to extract features from.",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=37,
        help="number of neighbors to extract.",
    )
    parser.add_argument(
        "--tile_size", type=int, default=224, help="size of the tiles to extract."
    )
    parser.add_argument("--use_gpu", action="store_true", help="whether to use gpu.")
    args = parser.parse_args()

    main(
        path_visium=args.path_visium,
        path_slide=args.path_slide,
        path_output_folder=args.path_output_folder,
        level=args.level,
        knn=args.knn,
        tile_size=args.tile_size,
        use_gpu=args.use_gpu,
    )
