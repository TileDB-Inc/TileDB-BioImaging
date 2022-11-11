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
        """
        Levels are numbered from 0 (highest resolution) to level_count - 1 (lowest resolution).

        :return: The number of levels in the slide
        """
        return cast(int, self._osd.level_count)

    def level_axes(self, level: int) -> Axes:
        """
        Axes of this level

        :param level: Number corresponding to a level

        :return: Axes object containing the axes members
        """
        return Axes("YXC")

    def level_image(self, level: int) -> np.ndarray:
        """
        The image of a resolution

        :param level: Number corresponding to a level

        :return: np.ndarray of the image on the level given
        """
        dims = self._osd.level_dimensions[level]
        # image is in (width, height, channel) == XYC
        image = self._osd.read_region((0, 0), level, dims).convert("RGB")
        # np.asarray() transposes it to (height, width, channel) == YXC
        # https://stackoverflow.com/questions/49084846/why-different-size-when-converting-pil-image-to-numpy-array
        return np.asarray(image)

    def level_metadata(self, level: int) -> Dict[str, Any]:
        """
        The metadata of a resolution

        :param level: Number corresponding to a level

        :return: A Dict containing the metadata of the given level
        """
        return {}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        """
        The metadata of a group of resolutions (whole image)

        :return: A Dict containing the metadata of the image
        """
        return {}


class OpenSlideConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OpenSlideReader(input_path)

    def _get_image_writer(self, output_path: str) -> ImageWriter:
        raise NotImplementedError
