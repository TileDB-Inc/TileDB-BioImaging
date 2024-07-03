from pathlib import Path

import random
import numpy as np
from skimage.metrics import structural_similarity

import tiledb
from tiledb.bioimg import ATTR_NAME
from tiledb.cc import WebpInputFormat
from tiledb.bioimg.helpers import merge_ned_ranges

DATA_DIR = Path(__file__).parent / "data"


def get_schema(x_size, y_size, c_size=3, compressor=tiledb.ZstdFilter(level=0)):
    dims = []
    x_tile = min(x_size, 1024)
    y_tile = min(y_size, 1024)
    # WEBP Compressor does not accept specific dtypes so for dimensions we use the default
    dim_compressor = tiledb.ZstdFilter(level=0)
    if not isinstance(compressor, tiledb.WebpFilter):
        dim_compressor = compressor
    if isinstance(compressor, tiledb.WebpFilter):
        x_size *= c_size
        x_tile *= c_size
        if compressor.input_format == WebpInputFormat.WEBP_NONE:
            if c_size == 3:
                input_format = WebpInputFormat.WEBP_RGB
            elif c_size == 4:
                input_format = WebpInputFormat.WEBP_RGBA
            else:
                assert False, f"No WebpInputFormat with pixel_depth={c_size}"
            compressor = tiledb.WebpFilter(
                input_format=input_format,
                quality=compressor.quality,
                lossless=compressor.lossless,
            )
    else:
        dims.append(
            tiledb.Dim(
                "C",
                (0, c_size - 1),
                tile=c_size,
                dtype=np.uint32,
                filters=tiledb.FilterList([compressor]),
            )
        )

    dims.append(
        tiledb.Dim(
            "Y",
            (0, y_size - 1),
            tile=y_tile,
            dtype=np.uint32,
            filters=tiledb.FilterList([dim_compressor]),
        )
    )
    dims.append(
        tiledb.Dim(
            "X",
            (0, x_size - 1),
            tile=x_tile,
            dtype=np.uint32,
            filters=tiledb.FilterList([dim_compressor]),
        )
    )

    return tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        attrs=[
            tiledb.Attr(
                name=ATTR_NAME, dtype=np.uint8, filters=tiledb.FilterList([compressor])
            )
        ],
    )


def get_path(uri):
    return DATA_DIR / uri


def assert_image_similarity(im1, im2, min_threshold=0.95, channel_axis=-1, win_size=11):
    s = structural_similarity(im1, im2, channel_axis=channel_axis, win_size=win_size)
    assert s >= min_threshold, (s, min_threshold, im1.shape)


def generate_test_case(num_axes, num_ranges, max_value):
    """
    Generate a test case with a given number of axes and ranges.

    Parameters:
    num_axes (int): Number of axes.
    num_ranges (int): Number of ranges.
    max_value (int): Maximum value for range endpoints.

    Returns:
    tuple: A tuple containing the generated input and the expected output.
    """
    input_ranges = []

    for _ in range(num_ranges):
        ranges = []
        for _ in range(num_axes):
            start = random.randint(0, max_value - 1)
            end = random.randint(start, max_value)
            ranges.append((start, end))
        input_ranges.append(tuple(ranges))

    input_ranges = tuple(input_ranges)

    expected_output = merge_ned_ranges(input_ranges)

    return input_ranges, expected_output
