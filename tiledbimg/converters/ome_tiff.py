from typing import Sequence

import numpy as np
import tifffile

from .base import ImageConverter, ImageReader


class OMETiffReader(ImageReader):
    def __init__(self, input_path: str):
        self._tiff_series = tifffile.TiffFile(input_path).series

    @property
    def level_count(self) -> int:
        return len(self._tiff_series)

    @property
    def level_downsamples(self) -> Sequence[float]:
        # TODO
        return ()

    def level_image(self, level: int) -> np.ndarray:
        return self._tiff_series[level].asarray().swapaxes(0, 2)


class OMETiffConverter(ImageConverter):
    """Converter of Tiff-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OMETiffReader(input_path)
