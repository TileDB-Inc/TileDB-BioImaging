from __future__ import annotations

from operator import itemgetter
from typing import Sequence, Tuple

import numpy as np

import tiledb

from .converters.axes import transpose_array


class TileDBOpenSlide:
    def __init__(self, level_info: Sequence[Tuple[str, int, int]]):
        self._level_uris = tuple(map(itemgetter(0), level_info))
        self._level_dims = tuple(map(itemgetter(1, 2), level_info))  # (width, height)

    @classmethod
    def from_group_uri(cls, uri: str) -> TileDBOpenSlide:
        """
        :param uri: uri of a tiledb.Group containing the image
        :return: A TileDBOpenSlide object
        """
        with tiledb.Group(uri) as G:
            level_info = []
            for o in G:
                uri = o.uri
                with tiledb.open(uri) as a:
                    level = a.meta.get("level", 0)
                    width = a.shape[-1]
                    height = a.shape[-2]
                    level_info.append((level, uri, width, height))
            # sort by level
            level_info.sort(key=itemgetter(0))
        return cls(tuple(map(itemgetter(1, 2, 3), level_info)))

    @property
    def level_count(self) -> int:
        """
        Levels are numbered from 0 (highest resolution)
        to level_count - 1 (lowest resolution).

        :return: The number of levels in the slide
        """
        return len(self._level_dims)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """A (width, height) tuple for level 0 of the slide."""
        return self.level_dimensions[0]

    @property
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        """
        A sequence of (width, height) tuples, one for each level of the slide.
        level_dimensions[k] are the dimensions of level k.

        :return: A sequence of dimensions for each level
        """
        return self._level_dims

    @property
    def level_downsamples(self) -> Sequence[float]:
        """
        A sequence of downsample factors for each level of the slide.
        level_downsamples[k] is the downsample factor of level k.
        """
        level_dims = self.level_dimensions
        l0_w, l0_h = level_dims[0]
        return tuple((l0_w / w + l0_h / h) / 2.0 for w, h in level_dims)

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
        dim_to_slice = {"X": slice(x, x + w), "Y": slice(y, y + h)}
        with tiledb.open(self._level_uris[level]) as a:
            dims = [dim.name for dim in a.domain]
            image = a[tuple(dim_to_slice.get(dim, slice(None)) for dim in dims)]
        # transpose image to YXC
        return transpose_array(image, dims, "YXC")

    def get_best_level_for_downsample(self, factor: float) -> int:
        """Return the best level for displaying the given downsample filtering by factor.

        :param factor: The factor of downsamples. Above this value downsamples are filtered out.

        :return: The number corresponding to a level
        """
        lla = np.where(np.array(self.level_downsamples) < factor)[0]
        return int(lla.max() if len(lla) > 0 else 0)
