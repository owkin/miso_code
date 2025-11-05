"""
Copyright (c) Owkin Inc.
This source code is licensed under the CC BY-NC-SA 4.0 license found in the
LICENSE file in the root directory of this source tree.

Script to extract features from a Visium sample and its associated slide.
"""
from __future__ import annotations

from pathlib import Path
import argparse
from collections.abc import Callable
from loguru import logger
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from tqdm import tqdm
import openslide
import pandas as pd
import numpy as np
import torch
from PIL import Image


def main(
    path_visium: str,
    path_slide: str,
    path_output_folder: str,
    level: int,
    tile_size: int = 224,
    use_gpu: bool = False,
):
    """Main function to extract features from a Visium sample and its associated slide.

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
    tile_size : int
        size of the tiles to extract, default is 224.
    use_gpu : bool
        whether to use gpu, default is False.
    """
    path_visium = Path(path_visium)
    path_slide = Path(path_slide)
    path_output_folder = Path(path_output_folder)

    # Load H0-mini
    login()
    model = timm.create_model(
        "hf-hub:bioptimus/H0-mini",
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model.eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    wsi = openslide.OpenSlide(path_slide)
    tissue_positions = pd.read_csv(path_visium / "spatial" / "tissue_positions.csv")
    tissue_positions = tissue_positions[tissue_positions["in_tissue"] == 1]

    np.save(
        path_output_folder / "barcodes.npy",
        tissue_positions["barcode"].values.astype(str),
    )

    coords = tissue_positions.loc[
        :, ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].values.astype(int)
    downsample = wsi.level_downsamples[level]
    offset = tile_size * downsample / 2
    coords = (coords - np.round(offset)).astype(int)

    feats, feats_subtile = [], []
    for coord_tile in tqdm(coords, total=len(coords)):
        tile = np.array(wsi.read_region(coord_tile, level, (tile_size, tile_size)))[
            :, :, :3
        ]

        with torch.no_grad():
            tile = Image.fromarray(tile)
            output = model(transform(tile).unsqueeze(0).to(device))  # (1, 261, 768)

        # Patch token features (1, 256, 768)
        feats_subtile_ = patch_token_features = output[:, model.num_prefix_tokens:].squeeze().cpu().numpy()
        # CLS token features (1, 768):
        feats_ = output[:, 0].squeeze().cpu().numpy()

        coords_subtile = np.array(
            [[(level, i * 14, j * 14) for i in range(16)] for j in range(16)],
            dtype=float
        )
        coords_subtile = np.reshape(coords_subtile, (-1, 3))
        coords_subtile[:, 1:3] *= downsample
        coords_subtile[:, 1:3] += coord_tile

        feats.append(
            np.concatenate(
                (
                    np.array([level]),
                    coord_tile,
                    feats_,
                ),
                axis=0,
            )[None, :]
        )

        feats_subtile.append(
            np.concatenate(
                (
                    coords_subtile,
                    feats_subtile_,
                ),
                axis=1,
            )[None, :]
        )

    feats = np.concatenate(feats, axis=0)
    feats_subtile = np.concatenate(feats_subtile, axis=0)

    np.save(path_output_folder / "features.npy", feats)
    np.save(path_output_folder / "features_subtile.npy", feats_subtile)


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
        "--tile_size", type=int, default=224, help="size of the tiles to extract."
    )
    parser.add_argument("--use_gpu", action="store_true", help="whether to use gpu.")
    args = parser.parse_args()

    main(
        path_visium=args.path_visium,
        path_slide=args.path_slide,
        path_output_folder=args.path_output_folder,
        level=args.level,
        tile_size=args.tile_size,
        use_gpu=args.use_gpu,
    )
