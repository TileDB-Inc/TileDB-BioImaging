from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, cast

import numpy as np
import openslide as osd
from numpy._typing import NDArray

from tiledb import Config, Ctx
from tiledb.cc import WebpInputFormat
from tiledb.highlevel import _get_ctx

from ..helpers import cache_filepath, get_logger_wrapper, is_remote_protocol, iter_color
from . import DEFAULT_SCRATCH_SPACE
from .axes import Axes
from .base import ImageConverterMixin


class OpenSlideReader:

    _logger: logging.Logger

    def __init__(
        self,
        input_path: str,
        logger: Optional[logging.Logger] = None,
        *,
        source_config: Optional[Config] = None,
        source_ctx: Optional[Ctx] = None,
        dest_config: Optional[Config] = None,
        dest_ctx: Optional[Ctx] = None,
        scratch_space: str = DEFAULT_SCRATCH_SPACE,
    ):
        """
        OpenSlide image reader
        :param input_path: The path to the OpenSlide image

        """
        self._source_ctx = _get_ctx(source_ctx, source_config)
        self._source_cfg = self._source_ctx.config()
        self._dest_ctx = _get_ctx(dest_ctx, dest_config)
        self._dest_cfg = self._dest_ctx.config()
        self._logger = get_logger_wrapper(False) if not logger else logger
        if is_remote_protocol(input_path):
            resolved_path = cache_filepath(
                input_path, source_config, source_ctx, self._logger, scratch_space
            )
        else:
            resolved_path = input_path
        self._osd = osd.OpenSlide(resolved_path)

    def __enter__(self) -> OpenSlideReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._osd.close()

    @property
    def source_ctx(self) -> Ctx:
        return self._source_ctx

    @property
    def dest_ctx(self) -> Ctx:
        return self._dest_ctx

    @property
    def logger(self) -> Optional[logging.Logger]:
        return self._logger

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

    def optimal_reader(
        self, level: int, max_workers: int = 0
    ) -> Optional[Iterator[Tuple[Tuple[slice, ...], NDArray[Any]]]]:
        return None


class OpenSlideConverter(ImageConverterMixin[OpenSlideReader, Any]):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    _ImageReaderType = OpenSlideReader
