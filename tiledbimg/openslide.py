from __future__ import annotations

from dataclasses import dataclass, field
from operator import itemgetter
from typing import Sequence, Tuple

import numpy as np
import tiledb


@dataclass(frozen=True)
class LevelInfo:
    uri: str = field(compare=False, repr=False)
    dimensions: Tuple[int, int]


@dataclass(frozen=True)
class TileDBOpenSlide:
    level_info: Sequence[LevelInfo]

    @classmethod
    def from_group_uri(cls, uri: str) -> TileDBOpenSlide:
        with tiledb.Group(uri) as G:
            level_info = []
            for o in G:
                with tiledb.open(o.uri) as a:
                    level = a.meta.get("level", 0)
                    level_info.append((level, LevelInfo(o.uri, a.schema.shape[:2])))
            # sort by level
            level_info.sort(key=itemgetter(0))
        return cls(tuple(map(itemgetter(1), level_info)))

    @property
    def level_count(self) -> int:
        """
        The number of levels in the slide. Levels are numbered from 0 (highest resolution)
        to level_count - 1 (lowest resolution).
        """
        return len(self.level_info)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """A (width, height) tuple for level 0 of the slide."""
        return self.level_info[0].dimensions

    @property
    def level_dimensions(self) -> Sequence[Tuple[int, int]]:
        """
        A sequence of (width, height) tuples, one for each level of the slide.
        level_dimensions[k] are the dimensions of level k.
        """
        return tuple(li.dimensions for li in self.level_info)

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
        """
        x, y = location
        w, h = size
        with tiledb.open(self.level_info[level].uri) as a:
            data = a[x : x + w, y : y + h]
        return data

    def get_best_level_for_downsample(self, factor: float) -> int:
        """Return the best level for displaying the given downsample."""
        lla = np.where(np.array(self.level_downsamples) < factor)[0]
        return int(lla.max() if len(lla) > 0 else 0)
