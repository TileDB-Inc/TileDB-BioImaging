from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import tiledb

Number = Union[int, float]


@dataclass(frozen=True)
class LevelInfo:
    uri: str = field(compare=False)
    level: int = field(compare=True)
    dimensions: Tuple[int, ...] = field(compare=True)

    @classmethod
    def from_array(cls, array: tiledb.Array, level: Optional[int] = None) -> LevelInfo:
        uri = array.uri
        if level is None:
            basename = os.path.splitext(os.path.basename(uri))[0]
            try:
                level = int(basename.split("_")[-1])
            except ValueError:
                raise ValueError(f"Invalid level uri: {uri}")
        return cls(uri, level, array.schema.shape)


@dataclass(frozen=True)
class TileDBOpenSlide:
    level_info: Sequence[LevelInfo]
    level_downsamples: Sequence[float]

    @classmethod
    def from_group_uri(cls, uri: str) -> TileDBOpenSlide:
        with tiledb.Group(uri) as G:
            level_info = []
            for a_uri in sorted(o.uri for o in G):
                with tiledb.open(a_uri) as a:
                    level_info.append(LevelInfo.from_array(a))
            level_downsamples = G.meta.get("level_downsamples", ())
        return cls(tuple(level_info), level_downsamples)

    @property
    def level_count(self) -> int:
        """
        The number of levels in the slide. Levels are numbered from 0 (highest resolution)
        to level_count - 1 (lowest resolution).
        """
        return len(self.level_info)

    @property
    def dimensions(self) -> Tuple[int, ...]:
        """A (width, height) tuple for level 0 of the slide."""
        return self.level_info[0].dimensions[:2]

    @property
    def level_dimensions(self) -> Sequence[Tuple[int, ...]]:
        """
        A sequence of (width, height) tuples, one for each level of the slide.
        level_dimensions[k] are the dimensions of level k.
        """
        return tuple(li.dimensions[:2] for li in self.level_info)

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Return an RGB image containing the contents of the specified region as NumPy array.

        :param location: (x, y) tuple giving the top left pixel in the level 0 reference frame
        :param level: the level number
        :param size: (width, height) tuple giving the region size
        """
        x, y = location
        w, h = size
        with tiledb.open(self.level_info[level].uri) as a:
            data = a[x : x + w, y : y + h]
        return data["rgb"]

    def get_best_level_for_downsample(self, factor: float) -> int:
        """Return the best level for displaying the given downsample."""
        lla = np.where(np.array(self.level_downsamples) < factor)[0]
        return int(lla.max() if len(lla) > 0 else 0)


@dataclass
class SlideInfo:
    factor: Number  # scale factor
    slide: TileDBOpenSlide
    _level_data_cache: MutableMapping[Any, Any]

    def __init__(
        self,
        factor: Number,
        slide: TileDBOpenSlide,
        level_cache: MutableMapping[Any, Any] = {},
    ) -> None:
        self.factor = factor
        self.slide = slide
        self._downsample_level = self.slide.get_best_level_for_downsample(factor)

        if level_cache:
            assert isinstance(level_cache, dict)
            self._level_data_cache = level_cache
        else:
            self._level_data_cache = dict()

    def __repr__(self) -> str:
        slide_dims = self.slide.level_dimensions
        return f"""SlideInfo(factor={self.factor}, slide_level_dims={slide_dims})"""

    @property
    def slide_number(self) -> int:
        return 1

    """
    Return image data for a given layer with memoization.
    """

    def read_level(self, level: int) -> np.ndarray:
        if level in self._level_data_cache:
            return self._level_data_cache[level]

        uri = self.slide.level_info[level].uri

        # FIXME don't re-open
        with tiledb.open(uri) as A:
            data = A[:]
            self._level_data_cache[level] = data
            return data

    """
    Read region
    """

    def read_region(
        self, xy: Sequence[int], level: int, wh: Sequence[int]
    ) -> np.ndarray:
        x, y = xy
        w, h = wh

        return self.slide.read_region((x, y), level, (w, h))

    """
    Returns the current target downsample *level*.
    """

    @property
    def target_downsample_level(self) -> Any:
        return self._downsample_level

    @classmethod
    def from_group_uri(cls, factor: Number, uri: str) -> SlideInfo:
        tdb_slide = TileDBOpenSlide.from_group_uri(uri)
        return cls(factor, tdb_slide)
