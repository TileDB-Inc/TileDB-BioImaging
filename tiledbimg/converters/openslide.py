import numpy as np
import openslide as osd
import tiledb

from .base import ImageConverter


class OpenSlideConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    def convert_image(
        self, input_path: str, output_group_path: str, level_min: int = 0
    ) -> None:
        tiledb.group_create(output_group_path)
        img = osd.OpenSlide(input_path)
        for level in range(level_min, img.level_count):
            dims = img.level_dimensions[level]
            data = img.read_region((0, 0), level, dims).convert("RGB")
            data = np.asarray(data).swapaxes(0, 1)
            data = np.ascontiguousarray(data)
            data = data.view(
                dtype=np.dtype([("", "uint8"), ("", "uint8"), ("", "uint8")])
            )

            uri = self.output_level_path(output_group_path, level)
            schema = self.create_schema(data.shape)
            tiledb.Array.create(uri, schema)
            with tiledb.open(uri, "w") as A:
                A[:] = data

        # Write group metadata
        with tiledb.Group(output_group_path, "w") as G:
            G.meta["original_filename"] = input_path
            G.meta["level_downsamples"] = img.level_downsamples
