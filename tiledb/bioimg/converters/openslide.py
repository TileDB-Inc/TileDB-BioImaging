from typing import Any, Dict, cast

import numpy as np
import openslide as osd

from .base import Axes, ImageConverter, ImageReader, ImageWriter


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
    def level_count(self) -> int:
        return cast(int, self._osd.level_count)

    def level_axes(self, level: int) -> Axes:
        return Axes("YXC")

    def level_image(self, level: int) -> np.ndarray:
        dims = self._osd.level_dimensions[level]
        # image is in (width, height, channel) == XYC
        image = self._osd.read_region((0, 0), level, dims).convert("RGB")
        # np.asarray() transposes it to (height, width, channel) == YXC
        # https://stackoverflow.com/questions/49084846/why-different-size-when-converting-pil-image-to-numpy-array
        return np.asarray(image)

    def level_metadata(self, level: int) -> Dict[str, Any]:
        return {}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        return {}


class OpenSlideConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OpenSlideReader(input_path)

    def _get_image_writer(self, output_path: str) -> ImageWriter:
        raise NotImplementedError
