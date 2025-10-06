"""Module used to load slides' features from a processed 10x Visium dataset."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

import torch


class DatasetSlide:
    def __init__(
        self,
        path_to_feats: str,
        path_to_counts: str,
        path_to_neighbors: str,
        path_to_gene_list: str | None,
        normalization: str,
        n_tiles: int = 5000
    ):
        self.path_to_feats = Path(path_to_feats)
        self.path_to_counts = Path(path_to_counts)
        self.path_to_neighbors = Path(path_to_neighbors)
        self.path_to_gene_list = path_to_gene_list
        if path_to_gene_list is not None:
            self.path_to_gene_list = Path(self.path_to_gene_list)

        self.normalization = normalization

        self.slide_ids = os.listdir(self.path_to_feats)
        self.slide_ids = [
            self.path_to_feats / file
            for file in self.slide_ids
            if os.path.isdir(self.path_to_feats / file)
        ]
        self.slide_names = [slide_id.name for slide_id in self.slide_ids]

        self.features = []
        self.reads = []
        self.neighbors = []
        list_gene_names = []
        self.barcodes = []
        for slide_id in self.slide_ids:
            slide_data = self.load_slide_data(
                slide_id=slide_id,
                path_to_feats=self.path_to_feats,
                path_to_counts=self.path_to_counts,
                path_to_neighbors=self.path_to_neighbors,
                path_to_gene_list=self.path_to_gene_list,
                normalization=self.normalization,
            )
            self.features.append(slide_data["features"])
            self.reads.append(slide_data["reads"])
            self.neighbors.append(slide_data["neighbors"])
            list_gene_names.append(slide_data["gene_names"])
            self.barcodes.append(slide_data["barcodes"])

        self.slide_names_list = np.concatenate(
            [
                [self.slide_names[i]] * len(feats)
                for i, feats in enumerate(self.features)
            ]
        )

        self.features = np.concatenate(self.features, axis=0)
        self.coords = self.features[:, :3]
        self.features = self.features[:, 3:]
        self.neighbors = np.concatenate(self.neighbors, axis=0)
        self.barcodes = np.concatenate(self.barcodes, axis=0)

        self.gene_names = self.get_common_gene_names(list_gene_names, path_to_gene_list)
        self.reads = self.harmonize_reads(self.reads, self.gene_names, list_gene_names)

        self.in_dim = self.features.shape[1]
        self.out_dim = self.reads.shape[1]
        self.n_tiles = n_tiles

    def __len__(self) -> int:
        return len(self.slide_names)

    def __getitem__(self, idx: int) -> dict:
        slide_idx = self.slide_names_list == self.slide_names[idx]

        feats = self.features[slide_idx, :]
        coords = self.coords[slide_idx, :]
        reads = self.reads[slide_idx, :]
        neighbors = self.neighbors[slide_idx, :]

        if self.n_tiles is not None:
            n_tiles_slide, n_feats = feats.shape
            _, n_genes = reads.shape
            _, n_neighbors = neighbors.shape
            if self.n_tiles > n_tiles_slide:
                feats = np.vstack(
                    (feats, np.zeros((self.n_tiles - n_tiles_slide, n_feats)))
                )
                coords = np.vstack(
                    (coords, np.zeros((self.n_tiles - n_tiles_slide, 3)))
                )
                reads = np.vstack(
                    (reads, np.zeros((self.n_tiles - n_tiles_slide, n_genes)))
                )
                neighbors = np.vstack(
                    (neighbors, np.zeros((self.n_tiles - n_tiles_slide, n_neighbors)))
                )
            elif self.n_tiles < n_tiles_slide:
                feats = feats[: self.n_tiles, :]  # Later add random choice
                coords = coords[: self.n_tiles, :]
                reads = reads[: self.n_tiles, :]
                neighbors = neighbors[:self.n_tiles]
        dic = {
            "features":  torch.Tensor(feats),
            "coords":  torch.Tensor(coords),
            "labels":  torch.Tensor(reads),
            "slide_id": self.slide_names[idx],
            "neighbors": torch.Tensor(neighbors).long(),
        }

        return dic

    @staticmethod
    def load_slide_data(
        slide_id: str,
        path_to_feats: str,
        path_to_counts: str,
        path_to_neighbors: str,
        path_to_gene_list: str | None,
        normalization: str,
    ) -> dict:
        """Load a slide's data from a processed 10x Visium dataset."""
        neighbors = np.load(
            path_to_neighbors / slide_id / "neighbors.npy", mmap_mode="r"
        )
        features = np.load(path_to_feats / slide_id / "features.npy", mmap_mode="r")
        features_barcodes = np.load(
            path_to_feats / slide_id / "barcodes.npy", mmap_mode="r"
        )

        reads = np.load(
            path_to_counts / slide_id / normalization / "reads.npy", mmap_mode="r"
        )
        gene_names = np.load(
            path_to_counts / slide_id / normalization / "gene_names.npy", allow_pickle=True
        )
        reads_barcodes = np.load(
            path_to_counts / slide_id / normalization / "barcodes.npy", mmap_mode="r"
        )

        if not len(features_barcodes) == len(reads_barcodes):
            l1_in_l2 = np.isin(reads_barcodes, features_barcodes)
            l2_in_l1 = np.isin(features_barcodes, reads_barcodes)
            reads_barcodes = reads_barcodes[l1_in_l2]
            features_barcodes = features_barcodes[l2_in_l1]
            reads = reads[l1_in_l2]
            features = features[l2_in_l1]

        reindex = pd.Series(np.arange(len(reads)), index=reads_barcodes).loc[features_barcodes].values
        reads = reads[reindex]
        reads_barcodes = reads_barcodes[reindex]
        if not np.all(features_barcodes == reads_barcodes):
            raise ValueError("Features' barcodes are not aligned on reads'.")

        return {
            "neighbors": neighbors,
            "features": features,
            "reads": reads,
            "gene_names": gene_names,
            "barcodes": features_barcodes,
        }

    @staticmethod
    def get_common_gene_names(
        gene_names: list[str], path_to_gene_list: Path | None
    ) -> list[str]:
        """Find the common list of genes sequenced in the dataset."""
        gene_names = list(
            set.intersection(*[set(gene_list) for gene_list in gene_names])
        )

        # Restricting the list of genes to a specified gene list
        if path_to_gene_list is not None:
            _gene_list = pd.read_csv(path_to_gene_list)
            _symbol_list = _gene_list["symbol"].values
            gene_names = list(set(list(_symbol_list)) & set(gene_names))

        gene_names.sort()

        return gene_names

    @staticmethod
    def harmonize_reads(
        reads: list[np.ndarray], gene_names: list[str], list_gene_names: list[list[str]]
    ) -> np.ndarray:
        reads_idx, intersect_idx, unique_idx = [], [], []
        for i, temp_gene_names in enumerate(list_gene_names.copy()):
            # Sorting reads by gene name's alphabetical order
            reads_idx.append(np.argsort(temp_gene_names))

            # Restricting reads output to gene sequenced in every slide of the dataset
            temp_gene_names = temp_gene_names[reads_idx[i]]
            intersect_idx.append(pd.Series(temp_gene_names).isin(gene_names).values)

            temp_gene_names = temp_gene_names[intersect_idx[i]]
            # Restricting reads output to unique gene
            unique_idx.append(np.unique(temp_gene_names, return_index=True)[1])
        labels = np.vstack(
            [
                slide_reads[:, reads_idx[i]][:, intersect_idx[i]][:, unique_idx[i]]
                for i, slide_reads in enumerate(reads)
            ]
        )
        return labels


class DatasetSlideSubsample:
    def __init__(
        self,
        path_to_feats: str,
        path_to_counts: str,
        path_to_gene_list: str | None,
        normalization: str,
    ):
        self.path_to_feats = Path(path_to_feats)
        self.path_to_counts = Path(path_to_counts)
        self.path_to_gene_list = path_to_gene_list
        if path_to_gene_list is not None:
            self.path_to_gene_list = Path(self.path_to_gene_list)

        self.normalization = normalization

        self.slide_ids = os.listdir(self.path_to_feats)
        self.slide_ids = [
            self.path_to_feats / file
            for file in self.slide_ids
            if os.path.isdir(self.path_to_feats / file)
        ]
        self.slide_names = [slide_id.name for slide_id in self.slide_ids]

        self.features = []
        self.reads = []
        self.neighbors = []
        list_gene_names = []
        self.barcodes = []
        for slide_id in self.slide_ids:
            slide_data = self.load_slide_data(
                slide_id=slide_id,
                path_to_feats=self.path_to_feats,
                path_to_counts=self.path_to_counts,
                path_to_gene_list=self.path_to_gene_list,
                normalization=self.normalization,
            )
            # self.features.append(slide_data["features"])
            self.reads.append(slide_data["reads"])
            list_gene_names.append(slide_data["gene_names"])
            self.barcodes.append(slide_data["barcodes"])

        self.slide_names = np.concatenate(
            [
                [self.slide_names[i]] * len(reads)
                for i, reads in enumerate(self.reads)
            ]
        )
        self.tile_idx = np.concatenate(
            [
                np.arange(len(reads))
                for reads in self.reads
            ]
        )

        self.barcodes = np.concatenate(self.barcodes, axis=0)

        self.gene_names = self.get_common_gene_names(list_gene_names, path_to_gene_list)
        self.reads = self.harmonize_reads(self.reads, self.gene_names, list_gene_names)

        sample = np.load(
            self.path_to_feats / self.slide_names[0] / "features_subtile.npy", mmap_mode="r"
        )
        self.in_dim = sample[0, 0].shape[0] - 3
        self.out_dim = self.reads.shape[1]

    def __len__(self) -> int:
        return len(self.slide_names)

    def __getitem__(self, idx: int) -> dict:

        slide_id = self.slide_names[idx]
        tile_idx = self.tile_idx[idx]
        features = np.load(
            self.path_to_feats / slide_id / "features_subtile.npy", mmap_mode="r"
        )
        feats = features[tile_idx, :, 3:]
        coords = features[tile_idx, :, :3]
        reads = self.reads[idx, :]

        dic = {
            "features": feats,
            "coords": coords,
            "labels": reads,
            "slide_id": self.slide_names[idx],
        }

        return dic

    @staticmethod
    def load_slide_data(
        slide_id: str,
        path_to_feats: str,
        path_to_counts: str,
        path_to_gene_list: str | None,
        normalization: str,
    ) -> dict:
        """Load a slide's data from a processed 10x Visium dataset."""
        features_barcodes = np.load(
            path_to_feats / slide_id / "barcodes.npy", mmap_mode="r"
        )

        reads = np.load(
            path_to_counts / slide_id / normalization / "reads.npy", mmap_mode="r"
        )
        gene_names = np.load(
            path_to_counts / slide_id / normalization / "gene_names.npy", allow_pickle=True
        )
        reads_barcodes = np.load(
            path_to_counts / slide_id / normalization / "barcodes.npy", mmap_mode="r"
        )

        if not len(features_barcodes) == len(reads_barcodes):
            l1_in_l2 = np.isin(reads_barcodes, features_barcodes)
            l2_in_l1 = np.isin(features_barcodes, reads_barcodes)
            reads_barcodes = reads_barcodes[l1_in_l2]
            features_barcodes = features_barcodes[l2_in_l1]
            reads = reads[l1_in_l2]
            features = features[l2_in_l1]

        reindex = pd.Series(np.arange(len(reads)), index=reads_barcodes).loc[features_barcodes].values
        reads = reads[reindex]
        reads_barcodes = reads_barcodes[reindex]
        if not np.all(features_barcodes == reads_barcodes):
            raise ValueError("Features' barcodes are not aligned on reads'.")

        return {
            "reads": reads,
            "gene_names": gene_names,
            "barcodes": features_barcodes,
        }

    @staticmethod
    def get_common_gene_names(
        gene_names: list[str], path_to_gene_list: Path | None
    ) -> list[str]:
        """Find the common list of genes sequenced in the dataset."""
        gene_names = list(
            set.intersection(*[set(gene_list) for gene_list in gene_names])
        )

        # Restricting the list of genes to a specified gene list
        if path_to_gene_list is not None:
            _gene_list = pd.read_csv(path_to_gene_list)
            _symbol_list = _gene_list["symbol"].values
            gene_names = list(set(list(_symbol_list)) & set(gene_names))

        gene_names.sort()

        return gene_names

    @staticmethod
    def harmonize_reads(
        reads: list[np.ndarray], gene_names: list[str], list_gene_names: list[list[str]]
    ) -> np.ndarray:
        reads_idx, intersect_idx, unique_idx = [], [], []
        for i, temp_gene_names in enumerate(list_gene_names.copy()):
            # Sorting reads by gene name's alphabetical order
            reads_idx.append(np.argsort(temp_gene_names))

            # Restricting reads output to gene sequenced in every slide of the dataset
            temp_gene_names = temp_gene_names[reads_idx[i]]
            intersect_idx.append(pd.Series(temp_gene_names).isin(gene_names).values)

            temp_gene_names = temp_gene_names[intersect_idx[i]]
            # Restricting reads output to unique gene
            unique_idx.append(np.unique(temp_gene_names, return_index=True)[1])
        labels = np.vstack(
            [
                slide_reads[:, reads_idx[i]][:, intersect_idx[i]][:, unique_idx[i]]
                for i, slide_reads in enumerate(reads)
            ]
        )
        return labels
