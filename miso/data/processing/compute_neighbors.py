"""
Copyright (c) Owkin Inc.
This source code is licensed under the CC BY-NC-SA 4.0 license found in the
LICENSE file in the root directory of this source tree.

Script to extract neighbors indexes using pre-extracted features.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.neighbors import kneighbors_graph


def main(
    path_feats: str,
    path_output_folder: str,
    knn: int,
):
    """Main function to extract neighbors indexes using pre-extracted features.

    Parameters
    ----------
    path_feats : str
        path to the visium sample's slide's extracted feats.
    path_output_folder : Path
        path where the new slide will be saved.
    knn : int
        number of neighbors to extract.
    """
    path_feats = Path(path_feats)
    path_output_folder = Path(path_output_folder)

    pos = np.load(path_feats)[:, 1:3]
    n_tiles, _ = pos.shape
    k_graph = kneighbors_graph(pos, n_neighbors=knn, include_self=True).toarray()
    neighbors_idx = k_graph.nonzero()[1].reshape(n_tiles, knn)
    save_path = path_output_folder / "neighbors.npy"
    np.save(save_path, neighbors_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_feats",
        type=str,
        required=True,
        help="path to visium sample's slide's extracted feats.",
    )
    parser.add_argument(
        "--path_output_folder",
        type=str,
        required=True,
        help="path where the new slide will be saved.",
    )
    parser.add_argument(
        "--knn",
        type=int,
        required=True,
        help="number of neighbors to extract.",
    )
    args = parser.parse_args()

    main(
        path_feats=args.path_feats,
        path_output_folder=args.path_output_folder,
        knn=args.knn,
    )
