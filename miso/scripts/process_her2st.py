import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
import argparse

from miso.data.processing.compute_neighbors import main as compute_neighbors


def main(
    path_dataset: str,
    use_gpu: bool = True,
    knn: int = 37
):
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

    path_dataset = Path(path_dataset)
    Path(path_dataset / f'processed_data/').mkdir(exist_ok=True)

    for fname in Path(path_dataset / 'images/HE/').glob('*.jpg'):

        name = fname.name.split('.')[0]
        logger.info(f'processing {name}')
        img = np.array(Image.open(fname))

        df_reads = pd.read_csv(
            path_dataset /
            f'count-matrices/{name}.tsv.gz',
            sep='\t',
            index_col=0
        )
        df_pos = pd.read_csv(
            path_dataset /
            f'spot-selections/{name}_selection.tsv.gz',
            sep='\t'
        )
        index = (df_pos['x'].astype(str) + 'x' + df_pos['y'].astype(str)).values
        df_pos['index'] = index
        index = np.intersect1d(index, df_reads.index)
        df_pos = df_pos.set_index('index').loc[index]
        df_reads = df_reads.loc[index]

        features = np.zeros((len(df_pos), 771))
        features[:, 1:3] = df_pos[['pixel_y', 'pixel_x']].values
        feats = []
        feats_subtile = []

        for _, row in tqdm(df_pos.iterrows()):
            x = int(row['pixel_y'])
            y = int(row['pixel_x'])
            X = Image.fromarray(img[x-112:x+112, y-112:y+112])
            with torch.no_grad():
                output = model(transform(X).unsqueeze(0).to(device))  # (1, 261, 768)
                # CLS token features (1, 768):
            feats_ = output[:, 0]
            feats.append(feats_.cpu().numpy())
            # Patch token features (1, 256, 768)
            feats_subtile_ = patch_token_features = output[
                :,
                model.num_prefix_tokens:
            ].squeeze().cpu().numpy()
            # Arbitrarily, we set the z coordinates to 0 since there is no level coordinate here.
            coords_subtile = np.array(
                [[(0, i * 14, j * 14) for j in range(16)] for i in range(16)]
            )
            coords_subtile = np.array([0, row['pixel_y'], row['pixel_x']]) +\
                np.reshape(coords_subtile, (-1, 3))
            feats_subtile.append(
                np.concatenate(
                    (
                        coords_subtile,
                        feats_subtile_,
                    ),
                    axis=1,
                )[None, :]
            )
        feats = np.concatenate(feats)
        features[:, 3:] = feats
        features_subtiles = np.concatenate(feats_subtile)

        Path(path_dataset / f'processed_data/{name}').mkdir(exist_ok=True)
        Path(path_dataset / f'processed_data/{name}/raw_reads').mkdir(exist_ok=True)
        np.save(
            path_dataset / f'processed_data/{name}/features.npy',
            features
        )
        np.save(
            path_dataset / f'processed_data/{name}/features_subtile.npy',
            features_subtiles
        )
        np.save(
            path_dataset / f'processed_data/{name}/barcodes.npy',
            df_pos.index.values.astype('bytes')
        )
        np.save(
            path_dataset / f'processed_data/{name}/raw_reads/reads.npy',
            df_reads.values
        )
        np.save(
            path_dataset / f'processed_data/{name}/raw_reads/gene_names.npy',
            df_reads.columns
        )
        np.save(
            path_dataset / f'processed_data/{name}/raw_reads/barcodes.npy',
            df_reads.index.values.astype('bytes')
        )

        compute_neighbors(
            path_feats=path_dataset / f'processed_data/{name}/features.npy',
            path_output_folder=path_dataset / f'processed_data/{name}',
            knn=3,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset", type=str, help="Path to HER2ST data.")
    parser.add_argument("--use_gpu", action="store_true", help="whether to use gpu.")
    parser.add_argument(
        "--knn",
        type=int,
        default=37,
        help="number of neighbors to extract.",
    )
    args = parser.parse_args()

    main(
        path_dataset=args.path_dataset,
        use_gpu=args.use_gpu,
        knn=args.knn,
    )
