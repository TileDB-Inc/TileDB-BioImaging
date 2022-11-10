from typing import Any, Dict, cast

import numpy as np
import openphi as op

from .base import Axes, ImageConverter, ImageReader, ImageWriter


class ISyntaxReader(ImageReader):
    def __init__(self, input_path: str):
        self._ophi = op.OpenPhi(input_path)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._ophi.close()

    @property
    def level_count(self) -> int:
        return cast(int, self._ophi.level_count)

    def level_axes(self, level: int) -> Axes:
        return Axes("YXC")

    def level_metadata(self, level: int) -> Dict[str, Any]:
        return {}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        return {}

    def level_image(self, level: int) -> np.ndarray:
        dims = self._ophi.level_dimensions[level]
        # image is in (width, height, channel) == XYC
        image = self._ophi.read_region((0, 0), level, dims).convert("RGB")
        # np.asarray() transposes it to (height, width, channel) == YXC
        # https://stackoverflow.com/questions/49084846/why-different-size-when-converting-pil-image-to-numpy-array
        return np.asarray(image)


class ISyntaxConverter(ImageConverter):
    """Converter of OpenSlide-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return ISyntaxReader(input_path)

    def _get_image_writer(self, input_path: str, output_path: str) -> ImageWriter:
        raise NotImplementedError
