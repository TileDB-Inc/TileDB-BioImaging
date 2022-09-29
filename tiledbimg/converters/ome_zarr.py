import os

import numpy as np
import tiledb
import zarr

from .base import ImageConverter


class OMEZarrConverter(ImageConverter):
    """Converter of Zarr-supported images to TileDB Groups of Arrays"""

    def convert_image(
        self, input_path: str, output_group_path: str, level_min: int = 0
    ) -> None:
        tiledb.group_create(output_group_path)
        zarray = zarr.open(input_path)
        level_count = len(zarray)
        uris = []
        for level in range(level_min, level_count):
            zarray_l0 = zarray[level][0]
            zyx_shape = tuple(zarray_l0.shape[i] for i in (1, 3, 4))
            data = np.asarray(zarray_l0).reshape(zyx_shape).swapaxes(0, 2)
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
