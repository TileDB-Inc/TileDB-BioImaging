import os

import numpy as np
import tiledb

from .base import ImageConverter

# outline
# - open file
# - collect shape information
# - calculate padded shapes
# - create ArraySchema


class OMEZarrConverter(ImageConverter):
    def convert_image(
        self, input_path: str, output_group_path: str, level_min: int = 0
    ) -> None:

        import zarr

        """
        Convert a Zarr-supported image to a TileDB Group of Arrays, one
        per level.

        :param input_path: path to the Zarr-supported image
        :param output_group_path: path to the TildDB group of arrays
        """

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
