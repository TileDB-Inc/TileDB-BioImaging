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
        uris = []
        for level in range(level_min, img.level_count):
            dims = img.level_dimensions[level]
            data = img.read_region((0, 0), level, dims).convert("RGB")
            data = np.asarray(data).swapaxes(0, 1)
            uris.append(self._write_level(output_group_path, level, data))
        self._write_metadata(output_group_path, input_path, img.level_downsamples, uris)
