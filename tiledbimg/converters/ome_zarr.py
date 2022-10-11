import numpy as np
import zarr

from .base import ImageConverter, ImageReader


class OMEZarrReader(ImageReader):
    def __init__(self, input_path: str):
        self._zarray = zarr.open(input_path)

    @property
    def level_count(self) -> int:
        return len(self._zarray)

    def level_image(self, level: int) -> np.ndarray:
        zarray_l0 = self._zarray[level][0]
        zyx_shape = tuple(zarray_l0.shape[i] for i in (1, 3, 4))
        reshaped = np.asarray(zarray_l0).reshape(zyx_shape)
        return reshaped.swapaxes(0, 2)


class OMEZarrConverter(ImageConverter):
    """Converter of Zarr-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OMEZarrReader(input_path)
