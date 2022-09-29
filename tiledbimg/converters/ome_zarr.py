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
        uris = []
        for level in range(level_min, len(zarray)):
            zarray_l0 = zarray[level][0]
            zyx_shape = tuple(zarray_l0.shape[i] for i in (1, 3, 4))
            data = np.asarray(zarray_l0).reshape(zyx_shape).swapaxes(0, 2)
            uris.append(self._write_level(output_group_path, level, data))
        # TODO: level_downsamples
        self._write_metadata(output_group_path, input_path, uris=uris)
