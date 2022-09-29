import os

import numpy as np
import tifffile
import tiledb

from .base import ImageConverter


class OMETiffConverter(ImageConverter):
    """Converter of Tiff-supported images to TileDB Groups of Arrays"""

    def convert_image(
        self, input_path: str, output_group_path: str, level_min: int = 0
    ) -> None:
        tiledb.group_create(output_group_path)
        tiff_series = tifffile.TiffFile(input_path).series
        level_count = len(tiff_series)
        uris = []
        for level in range(level_min, level_count):
            data = tiff_series[level].asarray().swapaxes(0, 2)
            data = np.ascontiguousarray(data)
            data = data.view(
                dtype=np.dtype([("", "uint8"), ("", "uint8"), ("", "uint8")])
            )

            uri = self.output_level_path(output_group_path, level)
            schema = self.create_schema(data.shape)
            tiledb.Array.create(uri, schema)
            with tiledb.open(uri, "w") as A:
                A[:] = data
            uris.append(uri)

        with tiledb.Group(output_group_path, "w") as G:
            G.meta["original_filename"] = input_path
            # TODO G.meta["level_downsamples"] = level_count
            for level_uri in uris:
                G.add(os.path.basename(level_uri), relative=True)
