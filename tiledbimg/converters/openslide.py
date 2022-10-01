from typing import Sequence, cast

import numpy as np
import openslide as osd

from .base import ImageConverter, ImageReader


class OpenSlideReader(ImageReader):
    def __init__(self, input_path: str):
        self._osd = osd.OpenSlide(input_path)

    @property
    def level_count(self) -> int:
        return cast(int, self._osd.level_count)

    @property
    def level_downsamples(self) -> Sequence[float]:
        return cast(Sequence[float], self._osd.level_downsamples)

    def level_image(self, level: int) -> np.ndarray:
        dims = self._osd.level_dimensions[level]
        image = self._osd.read_region((0, 0), level, dims).convert("RGB")
        return np.asarray(image).swapaxes(0, 1)


class OpenSlideConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OpenSlideReader(input_path)
