import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import numpy as np

from . import WIN_OPENSLIDE_PATH

if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(WIN_OPENSLIDE_PATH):
        import openslide as osd
else:
    import openslide as osd

from tiledb.cc import WebpInputFormat

from ..helpers import get_logger_wrapper, iter_color
from .axes import Axes
from .base import ImageConverter, ImageReader


class OpenSlideReader(ImageReader):
    def __init__(self, input_path: str, logger: Optional[logging.Logger] = None):
        """
        OpenSlide image reader

        :param input_path: The path to the OpenSlide image

        """
        self._logger = get_logger_wrapper(False) if not logger else logger
        self._osd = osd.OpenSlide(input_path)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._osd.close()

    @property
    def logger(self) -> Optional[logging.Logger]:
        return self._logger

    @logger.setter
    def logger(self, default_logger: logging.Logger) -> None:
        self._logger = default_logger

    @property
    def axes(self) -> Axes:
        axes = Axes("YXC")
        self._logger.debug(f"Reader axes: {axes}")
        return axes

    @property
    def channels(self) -> Sequence[str]:
        return "RED", "GREEN", "BLUE", "ALPHA"

    @property
    def webp_format(self) -> WebpInputFormat:
        self._logger.debug(f"Webp Input Format: {WebpInputFormat.WEBP_RGBA}")
        return WebpInputFormat.WEBP_RGBA

    @property
    def level_count(self) -> int:
        level_count = cast(int, self._osd.level_count)
        self._logger.debug(f"Level count: {level_count}")
        return level_count

    def level_dtype(self, level: int) -> np.dtype:
        dtype = np.dtype(np.uint8)
        self._logger.debug(f"Level {level} dtype: {dtype}")
        return dtype

    def level_shape(self, level: int) -> Tuple[int, ...]:
        width, height = self._osd.level_dimensions[level]
        # OpenSlide.read_region() returns a PIL image in RGBA mode
        # passing it to np.asarray() returns a (height, width, 4) array
        # https://stackoverflow.com/questions/49084846/why-different-size-when-converting-pil-image-to-numpy-array
        self._logger.debug(f"Level {level} shape: ({width}, {height}, 4)")
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
        self._logger.debug(f"Level {level} - Metadata: None")
        return {}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        self._logger.debug("Group metadata: None")
        return {}

    @property
    def image_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        color_generator = iter_color(np.dtype(np.uint8), 3)
        properties = self._osd.properties

        # We skip the alpha channel
        metadata["channels"] = [
            {"id": f"{idx}", "name": f"{name}", "color": next(color_generator)}
            for idx, name in enumerate(["red", "green", "blue"])
        ]

        if "aperio.MPP" in properties:
            metadata["physicalSizeX"] = metadata["physicalSizeY"] = float(
                properties["aperio.MPP"]
            )
            metadata["physicalSizeYUnit"] = metadata["physicalSizeYUnit"] = "µm"

        self._logger.debug(f"Image metadata: {metadata}")
        return metadata

    @property
    def original_metadata(self) -> Dict[str, Any]:
        return {"SVS": list(self._osd.properties.items())}


class OpenSlideConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    _ImageReaderType = OpenSlideReader
