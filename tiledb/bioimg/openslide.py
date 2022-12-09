from __future__ import annotations

from operator import itemgetter
from typing import Any, Iterator, Sequence, Tuple

import numpy as np

import tiledb

from .converters.axes import transpose_array


class TileDBOpenSlide:
    @classmethod
    def from_group_uri(cls, uri: str) -> TileDBOpenSlide:
        """
        :param uri: uri of a tiledb.Group containing the image
        :return: A TileDBOpenSlide object
        """
        with tiledb.Group(uri) as G:
            level_info = []
            for o in G:
                array = tiledb.open(o.uri)

                level = array.meta.get("level", 0)
                level_info.append((level, array))
            # sort by level
            level_info.sort(key=itemgetter(0))
        return cls(tuple(map(itemgetter(1), level_info)))

    def __init__(self, level_arrays: Sequence[tiledb.Array]):
        print(level_arrays)
        self._level_arrays = level_arrays

        self._channel_count = level_arrays[0]
        self._webp_compressed = (
            True
            if isinstance(level_arrays[0].attr(0).filters[0], tiledb.filter.WebpFilter)
            else False
        )

    def __enter__(self) -> TileDBOpenSlide:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for array in self._level_arrays:
            array.close()

    @property
    def channels(self) -> int:
        if isinstance(
            self._level_arrays[0].attr(0).filters[0], tiledb.filter.WebpFilter
        ):
            return 3
        else:
            assert self._level_arrays[0].schema.has_dim("C")
            return int(self._level_arrays[0].schema.domain.dim("C").domain[1]) + 1

    @property
    def level_count(self) -> int:
        """
        Levels are numbered from 0 (highest resolution)
        to level_count - 1 (lowest resolution).

        :return: The number of levels in the slide
        """
        return len(self._level_arrays)

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
        if self._webp_compressed:
            return tuple(
                (array.shape[-1] // self.channels, array.shape[-2])
                for array in self._level_arrays
            )
        else:
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
        array = self._level_arrays[level]
        dims = "".join(dim.name for dim in array.domain)

        if isinstance(array.attr(0).filters[0], tiledb.filter.WebpFilter):

            x, y = (location[0] * self.channels, location[1])
            w, h = (size[0] * self.channels, size[1])
        else:
            x, y = location
            w, h = size

        dim_to_slice = {"X": slice(x, x + w), "Y": slice(y, y + h)}
        image = array[tuple(dim_to_slice.get(dim, slice(None)) for dim in dims)]

        # print(f"Read shape {image.shape}")
        # image = np.reshape(image, (-1, image.shape[1] // 3, 3))
        # print(f"Reshaped shape {image.shape}")
        if isinstance(array.attr(0).filters[0], tiledb.filter.WebpFilter):
            return np.reshape(
                image, (-1, image.shape[1] // self.channels, self.channels)
            )
        else:
            # image = image.transpose((2, 0, 1))
            # transpose image to YXC
            return transpose_array(image, dims, "YXC")
        # return image

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
            yield a.shape[dims.index(a.dim("X"))], a.shape[dims.index(a.dim("Y"))]
