from typing import Any, Dict, Optional, Sequence, Tuple, cast

import numpy as np
import openslide as osd

from tiledb.cc import WebpInputFormat

from .axes import Axes
from .base import ImageConverter, ImageReader


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
    def channels(self) -> Sequence[str]:
        return "RED", "GREEN", "BLUE", "ALPHA"

    @property
    def webp_format(self) -> WebpInputFormat:
        return WebpInputFormat.WEBP_RGBA

    @property
    def level_count(self) -> int:
        return cast(int, self._osd.level_count)

    def level_dtype(self, level: int) -> np.dtype:
        return np.dtype(np.uint8)

    def level_shape(self, level: int) -> Tuple[int, ...]:
        width, height = self._osd.level_dimensions[level]
        # OpenSlide.read_region() returns a PIL image in RGBA mode
        # passing it to np.asarray() returns a (height, width, 4) array
        # https://stackoverflow.com/questions/49084846/why-different-size-when-converting-pil-image-to-numpy-array
        return height, width, 4

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
        return np.asarray(self._osd.read_region(location, level, size))

    def level_metadata(self, level: int) -> Dict[str, Any]:
        return {}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        return {}


class OpenSlideConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    _ImageReaderType = OpenSlideReader
