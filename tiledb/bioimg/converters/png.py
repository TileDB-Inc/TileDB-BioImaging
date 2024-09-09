import json
import logging
from functools import partial
from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy._typing import NDArray
from PIL import Image

from tiledb import VFS, Config, Ctx
from tiledb.cc import WebpInputFormat
from tiledb.highlevel import _get_ctx

from ..helpers import get_logger_wrapper, iter_color
from .axes import Axes
from .base import ImageConverter, ImageReader, ImageWriter


class PNGReader(ImageReader):
    def __init__(
        self,
        input_path: str,
        *,
        logger: Optional[logging.Logger] = None,
        source_config: Optional[Config] = None,
        source_ctx: Optional[Ctx] = None,
        dest_config: Optional[Config] = None,
        dest_ctx: Optional[Ctx] = None,
    ):

        self._logger = get_logger_wrapper(False) if not logger else logger
        self._input_path = input_path
        self._source_ctx = _get_ctx(source_ctx, source_config)
        self._source_cfg = self._source_ctx.config()
        self._dest_ctx = _get_ctx(dest_ctx, dest_config)
        self._dest_cfg = self._dest_ctx.config()
        self._vfs = VFS(config=self._source_cfg, ctx=self._source_ctx)
        self._vfs_fh = self._vfs.open(input_path, mode="rb")
        self._png = Image.open(self._vfs_fh)

        # Get initial metadata
        self._metadata: Dict[str, Any] = self._png.info
        self._metadata.update(dict(self._png.getexif()))

        # Handle all different modes as RGB for consistency
        if self._png.mode not in ["RGB", "RGBA"]:
            self._metadata["original_mode"] = self._png.mode
            self._png = self._png.convert("RGB")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._vfs.close(file=self._vfs_fh)

    @property
    def source_ctx(self) -> Ctx:
        return self._source_ctx

    @property
    def dest_ctx(self) -> Ctx:
        return self._dest_ctx

    @property
    def logger(self) -> Optional[logging.Logger]:
        return self._logger

    @logger.setter
    def logger(self, default_logger: logging.Logger) -> None:
        self._logger = default_logger

    @property
    def axes(self) -> Axes:
        axes = Axes(["X", "Y", "C"])
        self._logger.debug(f"Reader axes: {axes}")
        return axes

    @property
    def channels(self) -> Sequence[str]:
        if self.webp_format is WebpInputFormat.WEBP_RGB:
            self._logger.debug(f"Webp format: {WebpInputFormat.WEBP_RGB}")
            return "RED", "GREEN", "BLUE"
        self._logger.debug(f"Webp format is not: {WebpInputFormat.WEBP_RGB}")
        color_map = {"R": "RED", "G": "GREEN", "B": "BLUE"}
        # Use list comprehension to convert the short form to full form
        rgb_full = [color_map[color] for color in self._png.getbands()]
        return rgb_full

    @property
    def level_count(self) -> int:
        level_count = 1
        self._logger.debug(f"Level count: {level_count}")
        return level_count

    def level_dtype(self, level: int = 0) -> np.dtype:
        dtype = np.uint8
        self._logger.debug(f"Level {level} dtype: {dtype}")
        return dtype

    def level_shape(self, level: int = 0) -> Union[Tuple[int, ...], Any]:
        if level != 0:
            raise ValueError(
                "PNG images are not pyramidal and consist of only one level"
            )
        # Even after converting to RGB the size is not updated to 3d from 2d
        l_shape = (*self._png.size, 3)
        self._logger.debug(f"Level {level} shape: {l_shape}")
        return l_shape

    @property
    def webp_format(self) -> WebpInputFormat:
        self._logger.debug(f"Keyframe photometric: {self._png.mode}")
        if self._png.mode == "RGB":
            return WebpInputFormat.WEBP_RGB
        return WebpInputFormat.WEBP_NONE

    def level_image(
        self, level: int = 0, tile: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:

        if tile is None:
            return np.asarray(self._png)
        else:
            return np.asarray(self._png)[tile]

    def level_metadata(self, level: int) -> Dict[str, Any]:
        # Common with group metadata since there are no multiple levels
        writer_kwargs = dict(metadata=self._metadata)
        return {"json_write_kwargs": json.dumps(writer_kwargs)}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        writer_kwargs = dict(metadata=self._metadata)
        self._logger.debug(f"Group metadata: {writer_kwargs}")
        return {"json_write_kwargs": json.dumps(writer_kwargs)}

    @property
    def image_metadata(self) -> Dict[str, Any]:
        self._logger.debug(f"Image metadata: {self._metadata}")
        color_generator = iter_color(np.dtype(np.uint8), len(self.channels))

        channels = []
        for idx, channel in enumerate(self.channels):
            channel_metadata = {
                "id": f"{idx}",
                "name": f"Channel {idx}",
                "color": (next(color_generator)),
            }
            channels.append(channel_metadata)
        self._metadata["channels"] = channels
        return self._metadata

    @property
    def original_metadata(self) -> Dict[str, Any]:
        self._logger.debug(f"Original Image metadata: {self._metadata}")
        return self._metadata

    def optimal_reader(
        self, level: int, max_workers: int = 0
    ) -> Optional[Iterator[Tuple[Tuple[slice, ...], NDArray[Any]]]]:
        return None


class PNGWriter(ImageWriter):

    def __init__(self, output_path: str, logger: logging.Logger):
        self._logger = logger
        self._output_path = output_path
        self._group_metadata: Dict[str, Any] = {}
        self._writer = partial(Image.fromarray, mode="RGB")

    def compute_level_metadata(
        self,
        baseline: bool,
        num_levels: int,
        image_dtype: np.dtype,
        group_metadata: Mapping[str, Any],
        array_metadata: Mapping[str, Any],
        **writer_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:

        writer_metadata: Dict[str, Any] = {}
        original_mode = group_metadata.get("original_mode", "RGB")
        writer_metadata["mode"] = original_mode
        self._logger.debug(f"Writer metadata: {writer_metadata}")
        return writer_metadata

    def write_group_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._group_metadata = json.loads(metadata["json_write_kwargs"])

    def write_level_image(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> None:

        array_img = self._writer(image)
        original_img = array_img.convert(metadata["mode"])
        original_img.save(self._output_path, format="png")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class PNGConverter(ImageConverter):
    """Converter of Tiff-supported images to TileDB Groups of Arrays"""

    _ImageReaderType = PNGReader
    _ImageWriterType = PNGWriter
