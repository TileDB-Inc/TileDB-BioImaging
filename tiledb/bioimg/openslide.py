from __future__ import annotations

import warnings
from operator import attrgetter
from typing import Any, Mapping, MutableMapping, Sequence, Tuple, Union

import numpy as np

try:
    import dask.array as da
except ImportError:
    pass

import json

import tiledb
from tiledb import Config

from . import ATTR_NAME
from .converters.axes import Axes
from .helpers import open_bioimg


class TileDBOpenSlide:
    @classmethod
    def from_group_uri(cls, uri: str, attr: str = ATTR_NAME) -> TileDBOpenSlide:
        warnings.warn(
            "This method is deprecated, please use TileDBOpenSlide() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(uri, attr=attr)

    def __init__(self, uri: str, *, attr: str = ATTR_NAME, config: Config = None):
        """Open this TileDBOpenSlide.

        :param uri: uri of a tiledb.Group containing the image
        """
        self._config = config
        self._group = tiledb.Group(uri, config=config)
        pixel_depth = self._group.meta.get("pixel_depth", "")
        pixel_depth = dict(json.loads(pixel_depth)) if pixel_depth else {}
        self._levels = sorted(
            (
                TileDBOpenSlideLevel(o.uri, pixel_depth, attr=attr, config=config)
                for o in self._group
            ),
            key=attrgetter("level"),
        )

    def __enter__(self) -> TileDBOpenSlide:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for level in self._levels:
            level.close()
        self._group.close()

    @property
    def level_count(self) -> int:
        """Number of levels in the slide"""
        return len(self._levels)

    @property
    def levels(self) -> Sequence[int]:
        """Sequence of level numbers in the slide.

        Levels are numbered from `level_min` (highest resolution) to `level_count - 1`
        (lowest resolution), where `level_min` is the value of the respective
        `ImageConverter.to_tiledb` parameter (default=0) when creating the slide.
        """
        return tuple(map(attrgetter("level"), self._levels))

    @property
    def dimensions(self) -> Tuple[int, int]:
        """A (width, height) tuple for level 0 of the slide."""
        return self._levels[0].dimensions

    @property
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        """
        A sequence of (width, height) tuples, one for each level of the slide.
        level_dimensions[k] are the dimensions of level k.

        :return: A sequence of dimensions for each level
        """
        return tuple(map(attrgetter("dimensions"), self._levels))

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
        return self._levels[level].properties

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

    def _read_image(
        self,
        level: int,
        dim_slice: MutableMapping[str, slice] = {},
        to_original_axes: bool = False,
        to_dask: bool = False,
    ) -> Union[np.ndarray, da.Array]:
        axes = Axes(self._group.meta["axes"] if to_original_axes else "YXC")
        return self._levels[level].read(axes, dim_slice, to_dask)


class TileDBOpenSlideLevel:
    def __init__(
        self,
        uri: str,
        pixel_depth: Mapping[str, int],
        *,
        attr: str,
        config: Config = None,
    ):
        self._config = config
        self._tdb = open_bioimg(uri, attr=attr, config=config)
        self._pixel_depth = pixel_depth.get(str(self.level), 1)

    @property
    def level(self) -> int:
        return int(self._tdb.meta["level"])

    @property
    def dimensions(self) -> Tuple[int, int]:
        a = self._tdb
        dims = list(a.domain)
        width = a.shape[dims.index(a.dim("X"))]
        height = a.shape[dims.index(a.dim("Y"))]
        return width // self._pixel_depth, height

    @property
    def properties(self) -> Mapping[str, Any]:
        return dict(self._tdb.meta)

    def read(
        self,
        axes: Axes,
        dim_slice: MutableMapping[str, slice] = {},
        to_dask: bool = False,
    ) -> Union[np.ndarray, da.Array]:
        dims = tuple(dim.name for dim in self._tdb.domain)
        pixel_depth = self._pixel_depth
        if pixel_depth == 1:
            axes_mapper = axes.mapper(Axes(dims))
        else:
            x = dim_slice.get("X")
            if x is not None:
                dim_slice["X"] = slice(x.start * pixel_depth, x.stop * pixel_depth)
            axes_mapper = axes.webp_mapper(pixel_depth)

        array = da.from_tiledb(self._tdb) if to_dask else self._tdb
        selector = tuple(dim_slice.get(dim, slice(None)) for dim in dims)
        return axes_mapper.inverse.map_array(array[selector])

    def close(self) -> None:
        self._tdb.close()
