import os
from typing import Sequence

import numpy as np
import tifffile
import tiledb

from .base import ImageConverter

# outline
# - open file
# - collect shape information
# - calculate padded shapes
# - create ArraySchema


def page_shapes(f: tifffile.TiffFile) -> np.ndarray:

    """
    Opens a Tiff-supported image and returns shape information

    :param f: path to the the Tiff-supported image
    :return: NumPy array of the shapes
    """

    return [p.shape for p in f.pages]


def pad_width_to_tile(w: int, tile: int) -> int:

    """
    Reads the width and tile size and calculates padded width

    :param w: width
    :param tile: tile size
    :return: the calculated padded shape
    """

    return max([w + w % tile, tile])


def level_schema(
    shape: Sequence[int], tile_x: int = 1024, tile_y: int = 1024
) -> tiledb.ArraySchema:

    """
    Reads shape information, calculates padded shapes, and creates ArraySchema

    :param shape: input shape
    :param tile_x:
    :param tile_y:
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


class OMETiffConverter(ImageConverter):
    def convert_image(
        self, input_path: str, output_group_path: str, level_min: int = 0
    ) -> None:

        """
        Convert a Tiff-supported image to a TileDB Group of Arrays, one
        per level.

        :param input_path: path to the Tiff-supported image
        :param output_group_path: path to the TildDB group of arrays
        """

        tiff = tifffile.TiffFile(input_path)

        tiledb.group_create(output_group_path)

        level_count = len(tiff.series)
        uris = []
        for level in range(level_count)[level_min:]:
            # dims = tiff.series[level].shape

            output_img_path = self.output_level_path(output_group_path, level)

            slide_data = tiff.series[level].asarray().swapaxes(0, 2)
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


"""
data_path = "/Users/inorton/work/scratch/2022/0725-gt-histopath-demo/test1"
#img_path = os.path.join(data_path, "C3N-02572-22.svs")
img_path = os.path.join(data_path, "C3N-02572-22.ome.tiff")

import tempfile
output_path = tempfile.mkdtemp()
print("output path is: ", output_path)

cnv = GenericTiffConverter()

cnv.convert_image(img_path, output_path, level_min=0)
"""
