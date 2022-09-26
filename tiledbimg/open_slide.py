from dataclasses import dataclass
import tiledb
import numpy as np
import glob
import os

from pathlib import Path

from typing import Tuple, List, Dict
from numbers import Number

ATTRIBUTE_VALUE_NAME = "rgb"


@dataclass
class LevelInfo:
    level: int
    schema: tiledb.ArraySchema
    dimensions: Tuple[int]
    uri: str

    def __init__(self, abspath, level, schema):
        self.uri = abspath
        self.level = level
        self.schema = schema
        self.dimensions = schema.shape

    def __repr__(self) -> str:
        return f"""LevelInfo(level={self.level}, shape={self.schema.shape})"""

    @staticmethod
    def parse_level(s: str):
        exp = os.path.splitext(os.path.basename(s))[0]
        try:
            return int(exp.split("_")[-1])
        except Exception as exc:
            raise ValueError(f"Invalid level filename: {s}") from exc

    @classmethod
    def from_array(cls, path, level=None) -> "LevelInfo":
        if level is None:
            level = cls.parse_level(path)

        a = tiledb.open(path)

        return cls(path, level, a.schema)

    def __eq__(self, input):
        if not type(input) is LevelInfo:
            raise TypeError("Object types to compare should be the same")
        return self.level == input.level and self.dimensions == input.dimensions


class TileDBOpenSlide:
    _level_dimensions: List[Tuple[int]]
    _level_downsamples: List[float]
    _level_infos: List[LevelInfo]
    _group_metadat: Dict

    def __init__(
        self, level_infos, level_downsamples, level_dimensions, group_metadata
    ):

        self._level_infos = level_infos
        self._level_dimensions = level_dimensions
        self._level_downsamples = level_downsamples
        self._group_metadata = group_metadata

    def __eq__(self, x: "TileDBOpenSlide"):
        if not type(x) is TileDBOpenSlide:
            raise TypeError("Object types to compare should be the same")
        return (
            self.dimensions == x.dimensions
            and self._level_dimensions == x._level_dimensions
            and self._level_downsamples == x._level_downsamples
            and self._level_infos == x._level_infos
        )

    @classmethod
    def from_group_uri(cls, slide_group_uri, ctx=None):

        print(f"[DEBUG] slide_group_uri: {slide_group_uri}")
        breakpoint()
        with tiledb.Group(slide_group_uri) as G:
            group_meta = G.meta
            level_downsamples = G.meta.get("level_downsamples", None)
            group_dirs = [g.uri for g in G]

        level_infos = []
        group_dirs.sort()
        for a_uri in group_dirs:
            level_infos.append(LevelInfo.from_array(a_uri))

        level_dimensions = tuple(li.schema.shape[:2] for li in level_infos)

        return cls(level_infos, level_downsamples, level_dimensions, group_meta)

    """
    Returns the number of levels in this SlideTile array group.
    """

    @property
    def level_count(self):
        return len(self._level_dimensions)

    """
    Returns the dimensions of this SlideTile array group.
    """

    @property
    def level_dimensions(self):
        return self._level_dimensions

    """
    Reads given region of the specified level's image array.

    :param location: top-left starting position for region
    :param level: level number to slice from the SlideTile array group.
    :param size: region size (w, h)
    :return: NumPy array of the selected region
    """

    def read_region(self, xy, level, wh):
        x, y = xy
        w, h = wh

        uri = self._level_infos[level].uri

        with tiledb.open(uri) as A:
            data = A[x : x + w, y : y + h]
            data = data[ATTRIBUTE_VALUE_NAME]

        return data

    """
    Returns the highest resolution dimensions for the specified image array.
    """

    @property
    def dimensions(self):
        return self._level_infos[0].dimensions[:2]

    """
    Returns level
    """

    def get_best_level_for_downsample(self, factor: Number):
        lls = np.array(self._level_downsamples)
        lla = np.where(lls < factor)[0]
        return lla.max() if len(lla) > 0 else 0

    """
    Returns downsample factors for each image-array in the slide group.
    """

    @property
    def level_downsamples(self):
        return self._level_downsamples

    @property
    def level_info(self):
        return self._level_infos


@dataclass
class SlideInfo:
    factor: int  # scale factor
    slide: TileDBOpenSlide
    _level_data_cache: Dict

    def __init__(self, factor, slide, level_cache=None):
        self.factor = factor
        self.slide = slide
        self._downsample_level = self.slide.get_best_level_for_downsample(factor)

        if level_cache:
            assert isinstance(level_cache, dict)
            self._level_data_cache = level_cache
        else:
            self._level_data_cache = dict()

    def __repr__(self):
        slide_dims = self.slide.level_dimensions
        return f"""SlideInfo(factor={self.factor}, slide_level_dims={slide_dims})"""

    @property
    def slide_number(self):
        return 1

    """
    Return image data for a given layer with memoization.
    """

    def read_level(self, level: int):
        if level in self._level_data_cache:
            return self._level_data_cache[level]

        uri = self.slide._level_infos[level].uri

        # FIXME don't re-open
        with tiledb.open(uri) as A:
            data = A[:]
            self._level_data_cache[level] = data
            return data

    """
    Read region
    """

    def read_region(self, xy, level, wh):
        x, y = xy
        w, h = wh

        return self.slide.read_region((x, y), level, (w, h))

    """
    Returns the current target downsample *level*.
    """

    @property
    def target_downsample_level(self):
        return self._downsample_level

    @classmethod
    def from_group_uri(cls, factor, uri):
        tdb_slide = TileDBOpenSlide.from_group_uri(uri)
        return cls(factor, tdb_slide)
