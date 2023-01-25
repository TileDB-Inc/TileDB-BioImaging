from __future__ import annotations

import warnings
from typing import Any, Iterator, Mapping, MutableMapping, Sequence, Tuple, Union

import numpy as np

try:
    import dask.array as da
except ImportError:
    pass

import tiledb

from .converters.axes import Axes, AxesMapper


class TileDBOpenSlide:
    @classmethod
    def from_group_uri(cls, uri: str) -> TileDBOpenSlide:
        warnings.warn(
            "This method is deprecated, please use TileDBOpenSlide() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(uri)

    def __init__(self, uri: str):
        """Open this TileDBOpenSlide.

        :param uri: uri of a tiledb.Group containing the image
        """
        self._group = tiledb.Group(uri)
        self._level_arrays = [tiledb.open(o.uri) for o in self._group]
        # sort by level
        self._level_arrays.sort(key=lambda a: int(a.meta["level"]))

    def __enter__(self) -> TileDBOpenSlide:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for array in self._level_arrays:
            array.close()
        self._group.close()

    @property
    def level_count(self) -> int:
        """Number of levels in the slide"""
        return len(self._level_arrays)

    @property
    def levels(self) -> Sequence[int]:
        """Sequence of level numbers in the slide.

        Levels are numbered from `level_min` (highest resolution) to `level_count - 1`
        (lowest resolution), where `level_min` is the value of the respective
        `ImageConverter.to_tiledb` parameter (default=0) when creating the slide.
        """
        return tuple(a.meta["level"] for a in self._level_arrays)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """A (width, height) tuple for level 0 of the slide."""
        return next(self._iter_level_dimensions())

    @property
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        """
        A sequence of (width, height) tuples, one for each level of the slide.
        level_dimensions[k] are the dimensions of level k.

        :return: A sequence of dimensions for each level
        """
        return tuple(self._iter_level_dimensions())

    @property
    def level_downsamples(self) -> Sequence[float]:
        """
        A sequence of downsample factors for each level of the slide.
        level_downsamples[k] is the downsample factor of level k.
        """
        level_dims = self.level_dimensions
        l0_w, l0_h = level_dims[0]
        return tuple((l0_w / w + l0_h / h) / 2.0 for w, h in level_dims)

    @property
    def properties(self) -> Mapping[str, Any]:
        """Metadata about the slide"""
        return dict(self._group.meta)

    def level_properties(self, level: int) -> Mapping[str, Any]:
        """Metadata about the given slide level"""
        return dict(self._level_arrays[level].meta)

    def read_level(self, level: int, to_original_axes: bool = False) -> np.ndarray:
        """
        Return an image containing the contents of the specified level as NumPy array.

        :param level: the level number
        :param to_original_axes: If True return the image in the original axes,
            otherwise return it in YXC (height, width, channel) axes.
        """
        return self._read_image(level, to_original_axes=to_original_axes)

    def read_level_dask(self, level: int, to_original_axes: bool = False) -> da.Array:
        """
        Return an image containing the contents of the specified level as Dask array.

        :param level: the level number
        :param to_original_axes: If True return the image in the original axes,
            otherwise return it in YXC (height, width, channel) axes.
        """
        return self._read_image(level, to_original_axes=to_original_axes, to_dask=True)

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Return an image containing the contents of the specified region as NumPy array.

        :param location: (x, y) tuple giving the top left pixel in the level 0 reference frame
        :param level: the level number
        :param size: (width, height) tuple giving the region size

        :return: 3D (height, width, channel) Numpy array
        """
        x, y = location
        w, h = size
        return self._read_image(level, {"X": slice(x, x + w), "Y": slice(y, y + h)})

    def get_best_level_for_downsample(self, factor: float) -> int:
        """Return the best level for displaying the given downsample filtering by factor.

        :param factor: The factor of downsamples. Above this value downsamples are filtered out.

        :return: The number corresponding to a level
        """
        lla = np.where(np.array(self.level_downsamples) < factor)[0]
        return int(lla.max() if len(lla) > 0 else 0)

    def _iter_level_dimensions(self) -> Iterator[Tuple[int, int]]:
        for a in self._level_arrays:
            dims = list(a.domain)
            width = a.shape[dims.index(a.dim("X"))]
            height = a.shape[dims.index(a.dim("Y"))]
            yield width // get_pixel_depth(a), height

    def _read_image(
        self,
        level: int,
        dim_slice: MutableMapping[str, slice] = {},
        to_original_axes: bool = False,
        to_dask: bool = False,
    ) -> Union[np.ndarray, da.Array]:
        tdb = self._level_arrays[level]
        dims = tuple(dim.name for dim in tdb.domain)

        pixel_depth = get_pixel_depth(tdb)
        target_axes = Axes(self._group.meta["axes"] if to_original_axes else "YXC")
        if pixel_depth == 1:
            axes_mapper = AxesMapper(Axes(dims), target_axes)
        else:
            x = dim_slice.get("X")
            if x is not None:
                dim_slice["X"] = slice(x.start * pixel_depth, x.stop * pixel_depth)
            raise NotImplementedError

        array = da.from_tiledb(tdb) if to_dask else tdb
        selector = tuple(dim_slice.get(dim, slice(None)) for dim in dims)
        return axes_mapper.map_array(array[selector])


def get_pixel_depth(obj: Union[tiledb.Array, tiledb.Filter]) -> int:
    """Return the pixel depth for the given TileDB array or compression filter."""
    compressor = obj.attr(0).filters[0] if isinstance(obj, tiledb.Array) else obj
    if isinstance(compressor, tiledb.Filter):
        return 1
    raise TypeError(obj)
