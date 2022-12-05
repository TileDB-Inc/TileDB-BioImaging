from typing import Any, Dict, Tuple, cast

import numpy as np
import openslide as osd

from .base import Axes, ImageConverter, ImageReader


class OpenSlideReader(ImageReader):
    def __init__(self, input_path: str):
        """
        OpenSlide image reader

        :param input_path: The path to the OpenSlide image

        """
        self._osd = osd.OpenSlide(input_path)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._osd.close()

    @property
    def axes(self) -> Axes:
        return Axes("YXC")

    @property
    def level_count(self) -> int:
        return cast(int, self._osd.level_count)

    def level_dtype(self, level: int) -> np.dtype:
        return np.dtype(np.uint8)

    def level_shape(self, level: int) -> Tuple[int, ...]:
        width, height = self._osd.level_dimensions[level]
        # np.asarray() of a PIL image returns a (height, width, channel) array
        # https://stackoverflow.com/questions/49084846/why-different-size-when-converting-pil-image-to-numpy-array
        return height, width, 3

    def level_image(self, level: int) -> np.ndarray:
        return self._read_region(
            level, location=(0, 0), size=self._osd.level_dimensions[level]
        )

    def level_region(self, level: int, tile: Tuple[slice, ...]) -> np.ndarray:
        # tile: (Y slice, X slice, C slice)
        y, x, _ = tile
        return self._read_region(
            level,
            location=(x.start, y.start),
            size=(x.stop - x.start, y.stop - y.start),
        )

    def level_metadata(self, level: int) -> Dict[str, Any]:
        return {}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        return {}

    def _read_region(
        self, level: int, location: Tuple[int, int], size: Tuple[int, int]
    ) -> np.ndarray:
        return np.asarray(self._osd.read_region(location, level, size).convert("RGB"))


class OpenSlideConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    _ImageReaderType = OpenSlideReader
