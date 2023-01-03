from typing import Any, Dict, Optional, Tuple, cast

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

    def level_image(
        self, level: int, tile: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        level_size = self._osd.level_dimensions[level]
        if tile is None:
            location = (0, 0)
            size = level_size
        else:
            # tile: (Y slice, X slice, C slice)
            y, x, _ = tile
            full_size = self._osd.level_dimensions[0]
            # XXX: This is not 100% accurate if the level downsample factors are not integer
            # See https://github.com/openslide/openslide/issues/256
            location = (
                x.start * round(full_size[0] / level_size[0]),
                y.start * round(full_size[1] / level_size[1]),
            )
            size = (x.stop - x.start, y.stop - y.start)
        return np.asarray(self._osd.read_region(location, level, size).convert("RGB"))

    def level_metadata(self, level: int) -> Dict[str, Any]:
        return {}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        return {}


class OpenSlideConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    _ImageReaderType = OpenSlideReader
