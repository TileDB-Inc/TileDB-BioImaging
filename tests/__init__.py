import multiprocessing
import os
import urllib.parse
from typing import Sequence

import boto3
import numpy as np
import tiledb

if os.name == "posix":
    multiprocessing.set_start_method("forkserver")


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def rgb_to_5d(pixels: np.ndarray) -> Sequence:
    """Convert an RGB image into 5D image (t, c, z, y, x)."""
    if len(pixels.shape) == 2:
        stack = np.array([pixels])
        channels = np.array([stack])
    elif len(pixels.shape) == 3:
        size_c = pixels.shape[2]
        # Swapaxes for interchange x and y
        channels = [np.array([pixels[:, :, c].swapaxes(0, 1)]) for c in range(size_c)]
    else:
        assert False, f"expecting 2 or 3d: ({pixels.shape})"
    video = np.array([channels])
    return video


def get_CMU_1_SMALL_REGION_schemas():
    _domains = [
        ((0, 2219, 1024), (0, 2966, 1024), (0, 2, 3)),
        ((0, 386, 387), (0, 462, 463), (0, 2, 3)),
        ((0, 1279, 1024), (0, 430, 431), (0, 2, 3)),
    ]
    return [
        tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(
                    name="X",
                    domain=_domains[elem_id][0][:2],
                    tile=_domains[elem_id][0][2],
                    dtype=np.uint32,
                ),
                tiledb.Dim(
                    name="Y",
                    domain=_domains[elem_id][1][:2],
                    tile=_domains[elem_id][1][2],
                    dtype=np.uint32,
                ),
                tiledb.Dim(
                    name="C",
                    domain=_domains[elem_id][2][:2],
                    tile=_domains[elem_id][2][2],
                    dtype=np.uint32,
                ),
            ),
            sparse=False,
            attrs=[
                tiledb.Attr(
                    name="",
                    dtype=np.uint8,
                    filters=tiledb.FilterList([tiledb.ZstdFilter(level=0)]),
                )
            ],
            cell_order="row-major",
            tile_order="row-major",
            capacity=10000,
        )
        for elem_id in range(len(_domains))
    ]


def check_level_info(level, level_info):
    assert level_info.level == level
    assert tiledb.object_type(level_info.uri) == "array"
    assert isinstance(level_info.dimensions, tuple)
    assert all(isinstance(dim, int) for dim in level_info.dimensions)


def get_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def download_from_s3(uri: str) -> str:
    s3 = boto3.client("s3")
    parsed_uri = urllib.parse.urlparse(uri)
    local_path = get_path(os.path.basename(parsed_uri.path))
    if not os.path.exists(local_path):
        s3.download_file(parsed_uri.netloc, parsed_uri.path.lstrip("/"), local_path)
    return local_path
