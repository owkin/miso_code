"""
Copyright (c) Owkin Inc.
This source code is licensed under the CC BY-NC-SA 4.0 license found in the
LICENSE file in the root directory of this source tree.

Script to run super-resolved inference on a new slide."""

from __future__ import annotations

from typing import List
from pathlib import Path
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import openslide
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from miso.models import SimpleMIL


def run_super_resolved_inference(
    model: SimpleMIL,
    slide: openslide.OpenSlide,
    coords: np.array,
    gene_names: List,
    tile_size: int = 224,
    stride: int = 112,
    device: Literal['cuda' | 'cpu'] = 'cuda',
):
    """Run super-resolved inference on a new slide.

    Parameters
    ----------
    model : SimpleMIL
        Model prediction gene expression from features.
    slide : openslide.OpenSlide
    coords : np.array
        Numpy array with shape (n_tiles, 3) containing coordinates (level, x, y) of tiles
    gene_names : list
        List of genes predicted by the model
    stride : int
        Integer controling the overlap between tiles used for inference.
        Smaller values will give a smoother result, at the expanse of a longer computation.
    device : 'cuda' or 'cpu'
    """

    assert stride > 0, "Stride must be strictly positive"
    assert tile_size % stride == 0, "Tile size must be dividible by stride."

    login()
    extractor = timm.create_model(
        "hf-hub:bioptimus/H0-mini",
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    transform = create_transform(**resolve_data_config(extractor.pretrained_cfg, model=extractor))

    dz = DeepZoomGenerator(slide, tile_size, tile_size - stride)

    preds_patches = []
    
    for coord_tile in tqdm(coords, total=len(coords)):
        tile = dz.get_tile(int(coord_tile[0]), (coord_tile[1], coord_tile[2]))
    
        subtiles = []
        coords_subtiles = []

        # Divide the tile into subtiles with overlap
        for i in range(int(224 / stride)):
    
            for j in range(int(224 / stride)):
    
                subtile = np.array(tile)[
                    i * stride: i * stride + tile_size,
                    j * stride: j * stride + tile_size
                ]
                subtile = Image.fromarray(subtile)
                subtiles.append(transform(subtile).unsqueeze(0).to(device))
                
                coords_subtile = np.array(
                    [[(coord_tile[0], k * 14, l * 14) for k in range(16)] for l in range(16)],
                    dtype=float
                )
                coords_subtile = np.reshape(coords_subtile, (-1, 3))
                # project patch coordinates in the absolute coordinate system in pixel
                coords_subtile = coords_subtile + [
                    0,
                    ((j + 1) * stride - tile_size),
                    ((i + 1) * stride - tile_size)
                ]
                coords_subtile[:, 1:3] += coord_tile[1:3] * tile_size
                coords_subtiles.append(coords_subtile)

        # Extract features and predict with MISO
        with torch.no_grad():
            output = extractor(torch.cat(subtiles).to(device))
            pred = model.mlp(output[:, extractor.num_prefix_tokens:].squeeze()).cpu().numpy()

        # Store predictions and patch coordinates in a DataFrame
        df_tmp = pd.DataFrame(
            np.concatenate(
                (
                    np.concatenate(coords_subtiles).reshape(-1, 3),
                    pred.reshape(-1, len(gene_names)),
                ),
                axis=1,
            )
        )
        df_tmp.columns = ['z', 'x', 'y'] + list(gene_names)
        preds_patches.append(df_tmp)

    preds_patches = pd.concat(preds_patches)

    # Average multiple predictions for the same patch
    preds_patches = preds_patches.groupby(['x', 'y'], as_index=False).mean()

    return preds_patches
