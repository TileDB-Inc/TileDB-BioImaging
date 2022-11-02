from typing import cast

import numpy as np
import openslide as osd

from .base import ImageConverter, ImageReader, ImageWriter


class OpenSlideReader(ImageReader):
    def __init__(self, input_path: str):
        self._osd = osd.OpenSlide(input_path)

    @property
    def level_count(self) -> int:
        return cast(int, self._osd.level_count)

    def level_image(self, level: int) -> np.ndarray:
        dims = self._osd.level_dimensions[level]
        # image is in (width, height, channel) == XYC
        image = self._osd.read_region((0, 0), level, dims).convert("RGB")
        # np.asarray() transposes it to (height, width, channel) == YXC
        # https://stackoverflow.com/questions/49084846/why-different-size-when-converting-pil-image-to-numpy-array
        data = np.asarray(image)
        # we want (channel, height, width) so move channel first
        return np.moveaxis(data, 2, 0)


class OpenSlideConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OpenSlideReader(input_path)

    def _get_image_writer(self, input_path: str, output_path: str) -> ImageWriter:
        """Return an ImageWriter for the given input path."""
        pass
