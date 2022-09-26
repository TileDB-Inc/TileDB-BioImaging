import os
import sys

# import tifffile
# from tifffile import TiffFile
from typing import Sequence

import numpy as np
import tiledb
import zarr

from tiledbimg.util.common import ImageConverter

# outline
# - open file
# - collect shape information
# - calculate padded shapes
# - create ArraySchema


def page_shapes(f: zarr):

    """
    Opens a Zarr-supported image and returns shape information

    :param f: path to the the Zarr-supported image
    :return: NumPy array of the shapes
    """

    return [p.shape for p in f.pages]


def pad_width_to_tile(w, tile):

    """
    Reads the width and tile size and alculates padded width

    :param w: width
    :param tile: tile size
    :return: the calculated padded shape
    """

    return np.max([w + w % tile, tile])


def level_schema(shape: Sequence[int], tile_x=1024, tile_y=1024):

    """
    Reads shape information, calculates padded shapes, and creates ArraySchema

    :param shape: input shape
    :return: TileDB array
    """

    xmax = pad_width_to_tile(shape[0], tile_x)
    ymax = pad_width_to_tile(shape[1], tile_y)
    dims = [
        tiledb.Dim("X", domain=(0, xmax), tile=tile_x, dtype=np.uint64),
        tiledb.Dim("Y", domain=(0, ymax), tile=tile_y, dtype=np.uint64),
    ]
    domain = tiledb.Domain(*dims)
    attr = tiledb.Attr(
        name="",
        dtype=np.uint16,
        filters=tiledb.FilterList(
            [
                tiledb.ZstdFilter(level=7),
            ]
        ),
    )
    schema = tiledb.ArraySchema(attrs=[attr], domain=domain, sparse=False)
    return schema


class OMEZarrConverter(ImageConverter):
    def convert_image(self, input_path, output_group_path, level_min=0):

        import zarr

        """
        Convert a Zarr-supported image to a TileDB Group of Arrays, one
        per level.

        :param input_path: path to the Zarr-supported image
        :param output_group_path: path to the TildDB group of arrays
        """

        # tiff = tifffile.TiffFile(input_path)
        zarr = zarr.open(input_path)
        level_count = len(zarr)

        zarr_shape_x = np.zeros(level_count)
        zarr_shape_y = np.zeros(level_count)
        zarr_shape_z = np.zeros(level_count)

        for level in range(level_count):

            zarr_shape_x[level] = zarr[level][0].shape[4]
            zarr_shape_y[level] = zarr[level][0].shape[3]
            zarr_shape_z[level] = zarr[level][0].shape[1]

        tiledb.group_create(output_group_path)

        uris = []

        for level in range(level_count)[level_min:]:
            # dims = tiff.series[level].shape

            output_img_path = self.output_level_path(output_group_path, level)

            # slide_data = tiff.series[level].asarray().swapaxes(0, 2)
            slide_data = (
                np.asarray(zarr[level][0])
                .reshape(
                    zarr_shape_z[level].astype(int),
                    zarr_shape_y[level].astype(int),
                    zarr_shape_x[level].astype(int),
                )
                .swapaxes(0, 2)
            )
            data = np.ascontiguousarray(slide_data)
            newdata = data.view(
                dtype=np.dtype([("", "uint8"), ("", "uint8"), ("", "uint8")])
            )

            schema = self.create_schema(newdata.shape)
            tiledb.Array.create(output_img_path, schema)

            with tiledb.open(output_img_path, "w") as A:
                A[:] = newdata

            uris.append(output_img_path)

        with tiledb.Group(output_group_path, "w") as G:
            G.meta["original_filename"] = input_path
            # TODO G.meta["level_downsamples"] = level_count

            for level_uri in uris:
                level_subdir = os.path.basename(level_uri)
                G.add(level_subdir, relative=True)
