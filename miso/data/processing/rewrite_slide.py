"""Script used to reformat slide images to pyramidal wsi files."""

import os
import json
import time
from itertools import product
from pathlib import Path
from typing import Optional, Tuple
import argparse

import numpy as np
import tifffile
from loguru import logger
from numba import njit, prange
from numba.types import boolean, int_, uint8
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def get_spot_size(path_to_dir: Path) -> float:
    """Get the diameter in pixels of spots at full resolution.

    Arguments
    ---------
    path_to_dir : Path
        Path to the directory containing the scalefactors.json file.

    Returns
    -------
    float
        Diameter in pixels of spots.
    """
    path_to_scalefactors = list(path_to_dir.rglob("*.json"))[0]
    with open(path_to_scalefactors) as file:
        dic = json.load(file)
    return dic["spot_diameter_fullres"]


@njit(uint8[:, :, :](uint8[:, :, :], int_, boolean), parallel=True)
def integer_mean_pool(x: np.ndarray, downsample: int = 2, inplace=False) -> np.ndarray:
    """Naive k-k mean pool operation to be run on large integer arrays.

    If the input size is not a multiple of `downsample`, the output array
    is simply shrinked (equivalent of "valid" in torch or tf)

    It keeps memory allocation very low and is hence adapted for big numpy arrays.

    Timings: 900ms to downscale a RGB 30,000 x 30,000 pixels array.

    Note:
    1. A regular PIL.Image.resize would eat four times more RAM and take ~5 times longer
    to perform the exact resizing.
    2. This function is **typed** so you should pay attention to each argument type.
    3. This function performs a ceiling average: the mean of 1 and 2 is 1, mean of 2 and
    6 is 4, etc

    example:
    ``` python3
    x = np.random.randint(low=0, high=255, size=(512, 512, 3)).astype(np.uint8)

    x_downsampled = integer_mean_pool(x, 2, False)

    # Will fail ! `x` is int64
    integer_mean_pool(x.astype(np.int64), 2, False)

    # Will also fail ! `x` has no channels
    integer_mean_pool(x[:, :, 0], 2, False)

    # Will also fail ! `downsample` is not an integer multiple of 2
    integer_mean_pool(x, 2.5, False)
    ```

    Author: Rémy Dubois

    Parameters
    ----------
    x : np.ndarray
        Input (h, w, c) numpy array. Must be uint8 dtype.
    downsample : int
        Downsampling factor, must be a power of 2, by default 2
    inplace : bool, optional
        Works but results in non contiguous array. Should be used with caution, by
        default False

    Returns
    -------
    np.ndarray
        The downsampled numpy array

    Raises
    ------
    ValueError
        If downsampling factor is not a power of 2.
    """
    exp = np.log(downsample) / np.log(2)

    if (exp % 1) != 0:
        raise ValueError("Downsample must be a power of 2")

    # We compute mean of `downsample x downsample` pixels
    # Number of elements to reduce is squared so bitshift is x2
    bitshift = int(exp * 2)

    # If no downsampling, skip
    if bitshift == 0:
        return x

    h, w, c = x.shape

    output_size = (h // downsample, w // downsample, c)

    if not inplace:
        target = np.empty(output_size, dtype=np.uint8)
    else:
        target = x[::downsample, ::downsample]

    for i in prange(output_size[0]):  # pylint: disable=not-an-iterable
        _i = downsample * i
        for j in prange(output_size[1]):  # pylint: disable=not-an-iterable
            _j = downsample * j
            for k in range(c):
                # Mean reduction without implying any array instanciation / allocation,
                # otherwise parallelization fails
                accu = 0
                for l in range(downsample):  # noqa
                    for m in range(downsample):
                        accu += x[_i + l, _j + m, k]

                # Bitwise operator to divide by four while conserving integer dtype
                result = accu >> bitshift

                # Allocate. Idk how numba handles casting here.
                target[i, j, k] = result

    return target


def write_np_array_as_tif(
    img_array: np.ndarray,
    file_path: str,
    input_image_mpp: float,
    subsamplings: Tuple[int] = (1, 2, 4, 8, 16, 32, 64),
    tile_size: Tuple[int] = (224, 224),
    compression_scheme: Optional[
        tifffile.TIFF.COMPRESSION
    ] = tifffile.TIFF.COMPRESSION.JPEG,
    quality: Optional[int] = 90,
) -> None:
    """Write a (h, w, c) numpy array as a .tif file.

    This is meant at replacing pyvips, which is terribly buggy. It takes around 2
    minutes 30 seconds to write a 120,000 x 100,000 pixels WSI (34GB in RAM) at levels
    (1, 2, 8, 16) with 90 JPEG quality.

    Note that this function does not allow to write levels of resolution which are not
    power of 2 - downsamples from the base level. In other words, if your slide is at
    MPP 0.3, and you want to write it at MPP 0.5, you need to do it **before** calling
    this func.

    Example usage:
    ``` python3
    from skimage.data import astronaut

    img_array = astronaut()

    write_np_array_as_tif(
        img_array=img_array,
        file_path="/tmp/astronaut.tif",
        input_image_mpp=1.,
        subsamplings=(1, 2, 4),
        quality=90
        )

    ```

    Author: Rémy Dubois

    Parameters
    ----------
    img_array : np.ndarray
        Input (h, w, c) image as a numpy array
    file_path : str
        Where to write the path
    input_image_mpp : float
        Micrometers per pixel, the image definition.
        **This is the definition of the img_array, not the definition you want to write
        your image with**
        The definition you want to write your image with should be specified through
        `subsamplings`.
        As a reminder: mpp 0.25 <=> 40,000 pixels per centimeter <=> x40 magnification
    subsamplings : Tuple[int], optional
        Tuple of downsampling factor. 1 means no downsampling, by default
        (1, 2, 4, 8, 16, 32)
        Note: this defines the levels of resolution.
    tile_size : Tuple[int], optional
        Written tile size, by default (224, 224)
    compression_scheme : Optional[tifffile.TIFF.COMPRESSION], optional
        Compression scheme, one of tifffile.TIFF.COMPRESSION, by default
        tifffile.TIFF.COMPRESSION.JPEG
    quality : Optional[int], optional
        Image quality: 100 is perfect, 0 is terrible, by default None, if the
        compression scheme is None as well.
    """
    if not isinstance(img_array, np.ndarray):
        raise ValueError("Input should be np array")
    if img_array.ndim != 3:
        raise ValueError("Image should be (h, w, 3)")
    if img_array.shape[-1] != 3:
        raise ValueError("Input image array should be RGB (3 channels)")
    if img_array.dtype != np.uint8:
        raise ValueError(
            f"Input array should be uint8-dtyped. Received {img_array.dtype}."
        )

    with tifffile.TiffWriter(
        file_path, bigtiff=True, imagej=False, ome=False
    ) as outtif:
        options = dict(
            photometric="rgb",
            tile=tile_size,
            compression=(compression_scheme, quality, {}),
            metadata={},
        )

        # Convert mpp to pixels per centimeters
        base_res = int(1 / input_image_mpp * 10_000)

        st = time.perf_counter()
        for i, subsampling in enumerate(
            tqdm(
                subsamplings, desc="Writting image levels…", leave=False, disable=False
            )
        ):
            if i == 0:
                data = integer_mean_pool(img_array, subsampling, inplace=False)
            else:
                # Do not downsample from level 0, just downsample from previous level
                incr_subsampling = int(subsampling / subsamplings[i - 1])
                data = integer_mean_pool(data, incr_subsampling, inplace=False)

            h, w, c = data.shape
            tile_h, tile_w = tile_size
            n_tiles = (h // tile_h) * (w // tile_w)
            tile_coordinates = product(range(0, h, tile_h), range(0, w, tile_w))
            pbar = tqdm(
                tile_coordinates,
                total=n_tiles,
                leave=False,
                desc=f"Writting level {i}…",
            )

            tiles_gen = (data[i : i + tile_h, j : j + tile_w] for i, j in pbar)
            tiles_gen_contig = map(np.ascontiguousarray, tiles_gen)

            outtif.write(
                tiles_gen_contig,
                subfiletype=1,
                resolution=(
                    base_res / subsampling,
                    base_res / subsampling,
                    "CENTIMETER",
                ),
                **options,
                shape=(h, w, c),
                dtype=np.uint8,
            )
        delta = time.perf_counter() - st

    logger.success(f"Wrote tif tile at {file_path} in {delta:.0} seconds")


def main(
    path_visium: str,
    path_slide: str,
    path_output_folder: str,
    subsamplings=(1, 2, 4, 8, 16),
    quality=100,
):
    """Main function to reformat a slide image to a pyramidal wsi file.

    Parameters
    ----------
    path_visium : str
        path to the visium sample's directory.
    path_slide : Path
        path to the associated slide.
    path_output_folder : Path
        path where the new slide will be saved.
    subsamplings : tuple
        subsample factors, by default (1, 2, 4, 8, 16)
    quality : int
        compression quality (the higher the better, 100 max), by default 100
    """
    path_visium = Path(path_visium)
    path_slide = Path(path_slide)
    path_output_folder = Path(path_output_folder)

    spot_size = get_spot_size(path_visium)
    mpp = 55 / spot_size  # For 10xVisium a spot has a 55um diameter
    logger.info(f"Found spot_size is {spot_size} pixels and mpp is {mpp}.")

    arr = np.asarray(Image.open(path_slide)).astype(np.uint8)

    os.makedirs(str(path_output_folder), exist_ok=False)

    write_np_array_as_tif(
        img_array=arr,
        file_path=path_output_folder / (path_slide.stem + "_pyr.tif"),
        input_image_mpp=mpp,
        subsamplings=subsamplings,
        quality=quality,
    )


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
    args = parser.parse_args()

    main(
        path_visium=args.path_visium,
        path_slide=args.path_slide,
        path_output_folder=args.path_output_folder,
    )
