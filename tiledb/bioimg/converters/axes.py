from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, MutableSequence, Sequence, Tuple

import numpy as np
from pyeditdistance.distance import levenshtein


class AxesMapper(ABC):
    @property
    @abstractmethod
    def inverse(self) -> AxesMapper:
        """The axes mapper that inverts the effect of this one"""

    @abstractmethod
    def map_array(self, a: np.ndarray) -> np.ndarray:
        """Return the transformed Numpy array"""

    def map_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return the shape of the transformed Numpy array."""
        mapped_shape = list(shape)
        self.transform_shape(mapped_shape)
        return tuple(mapped_shape)

    def map_tile(self, tile: Tuple[slice, ...]) -> Tuple[slice, ...]:
        """Return the tile for slicing the transformed Numpy array"""
        mapped_tile = list(tile)
        self.transform_tile(mapped_tile)
        return tuple(mapped_tile)

    def transform_shape(self, shape: MutableSequence[int]) -> None:
        """Transform the given shape in place"""
        self.transform_sequence(shape)

    def transform_tile(self, tile: MutableSequence[slice]) -> None:
        """Transform the given tile in place"""
        self.transform_sequence(tile)

    def transform_sequence(self, s: MutableSequence[Any]) -> None:
        """Transform the given mutable sequence in place"""
        # intentionally not decorated as @abstractmethod: subclasses may override
        # transform_shape and transform_tile instead
        raise NotImplementedError


@dataclass(frozen=True)
class Swap(AxesMapper):
    i: int
    j: int

    @property
    def inverse(self) -> AxesMapper:
        return self

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return np.swapaxes(a, self.i, self.j)

    def transform_sequence(self, s: MutableSequence[Any]) -> None:
        i, j = self.i, self.j
        s[i], s[j] = s[j], s[i]


@dataclass(frozen=True)
class Move(AxesMapper):
    i: int
    j: int

    @property
    def inverse(self) -> AxesMapper:
        return Move(self.j, self.i)

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return np.moveaxis(a, self.i, self.j)

    def transform_sequence(self, s: MutableSequence[Any]) -> None:
        s.insert(self.j, s.pop(self.i))


@dataclass(frozen=True)
class Squeeze(AxesMapper):
    idxs: Tuple[int, ...]

    @property
    def inverse(self) -> AxesMapper:
        return Unsqueeze(self.idxs)

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return np.squeeze(a, self.idxs)

    def transform_sequence(self, s: MutableSequence[Any]) -> None:
        for i in sorted(self.idxs, reverse=True):
            del s[i]


@dataclass(frozen=True)
class Unsqueeze(AxesMapper):
    idxs: Tuple[int, ...]

    @property
    def inverse(self) -> AxesMapper:
        return Squeeze(self.idxs)

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return np.expand_dims(a, self.idxs)

    def transform_shape(self, shape: MutableSequence[int]) -> None:
        self.transform_sequence(shape, fill_value=1)

    def transform_tile(self, tile: MutableSequence[slice]) -> None:
        self.transform_sequence(tile, fill_value=slice(None))

    def transform_sequence(
        self, sequence: MutableSequence[Any], fill_value: Any = None
    ) -> None:
        for i in sorted(self.idxs):
            sequence.insert(i, fill_value)


@dataclass(frozen=True)
class YXC_TO_YX(AxesMapper):
    c_size: int

    @property
    def inverse(self) -> AxesMapper:
        return YX_TO_YXC(self.c_size)

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return a.reshape(self.map_shape(a.shape))

    def transform_shape(self, shape: MutableSequence[int]) -> None:
        y, x, c = shape
        if c != self.c_size:
            raise ValueError(f"C dimension must have size {self.c_size}: {c} given")
        shape[1] *= c
        del shape[2]

    def transform_tile(self, tile: MutableSequence[slice]) -> None:
        y, x, c = tile
        if c != slice(0, self.c_size):
            raise ValueError(
                f"C dimension can cannot be sliced: {c} given - {slice(0, self.c_size)} expected"
            )
        tile[1] = slice(x.start * self.c_size, x.stop * self.c_size)
        del tile[2]


@dataclass(frozen=True)
class YX_TO_YXC(AxesMapper):
    c_size: int

    @property
    def inverse(self) -> AxesMapper:
        return YXC_TO_YX(self.c_size)

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return a.reshape(self.map_shape(a.shape))

    def transform_shape(self, shape: MutableSequence[int]) -> None:
        c = self.c_size
        shape[1] //= c
        shape.append(c)

    def transform_tile(self, tile: MutableSequence[slice]) -> None:
        c = self.c_size
        tile[1] = slice(tile[1].start // c, tile[1].stop // c)
        tile.append(slice(0, c))


@dataclass(frozen=True)
class CompositeAxesMapper(AxesMapper):
    mappers: Sequence[AxesMapper]

    @property
    def inverse(self) -> AxesMapper:
        return CompositeAxesMapper([t.inverse for t in reversed(self.mappers)])

    def map_array(self, a: np.ndarray) -> np.ndarray:
        for mapper in self.mappers:
            a = mapper.map_array(a)
        return a

    def transform_shape(self, shape: MutableSequence[int]) -> None:
        for mapper in self.mappers:
            mapper.transform_shape(shape)

    def transform_tile(self, tile: MutableSequence[slice]) -> None:
        for mapper in self.mappers:
            mapper.transform_tile(tile)

    def transform_sequence(self, s: MutableSequence[Any]) -> None:
        for mapper in self.mappers:
            mapper.transform_sequence(s)


@dataclass(frozen=True)
class Axes:
    dims: str
    __slots__ = ("dims",)
    CANONICAL_DIMS = "TCZYX"

    def __init__(self, dims: Iterable[str]):
        if not isinstance(dims, str):
            dims = "".join(dims)
        axes = set(dims)
        if len(dims) != len(axes):
            raise ValueError(f"Duplicate axes: {dims}")
        for required_axis in "X", "Y":
            if required_axis not in axes:
                raise ValueError(f"Missing required axis {required_axis!r}")
        axes.difference_update(self.CANONICAL_DIMS)
        if axes:
            raise ValueError(f"{axes.pop()!r} is not a valid Axis")
        object.__setattr__(self, "dims", dims)

    def canonical(self, shape: Tuple[int, ...]) -> Axes:
        """
        Return a new Axes instance with the dimensions of this axes whose size in `shape`
        are greater than 1 and ordered in canonical order (TCZYX)
        """
        assert len(self.dims) == len(shape)
        dims = frozenset(dim for dim, size in zip(self.dims, shape) if size > 1)
        return Axes(dim for dim in self.CANONICAL_DIMS if dim in dims)

    def mapper(self, other: Axes) -> AxesMapper:
        """Return an AxesMapper from this axes to other"""
        return CompositeAxesMapper(list(_iter_axes_mappers(self.dims, other.dims)))

    def webp_mapper(self, num_channels: int) -> AxesMapper:
        """Return an AxesMapper from this 3D axes (YXC or a permutation) to 2D (YX)"""
        mappers = list(_iter_axes_mappers(self.dims, "YXC"))
        mappers.append(YXC_TO_YX(num_channels))
        return CompositeAxesMapper(mappers)


def _iter_axes_mappers(s: str, t: str) -> Iterator[AxesMapper]:
    s_set = frozenset(s)
    assert len(s_set) == len(s), f"{s!r} contains duplicates"
    t_set = frozenset(t)
    assert len(t_set) == len(t), f"{t!r} contains duplicates"

    common, squeeze_axes = [], []
    for i, m in enumerate(s):
        if m in t_set:
            common.append(m)
        else:
            squeeze_axes.append(i)
    if squeeze_axes:
        # source has extra dims: squeeze them
        yield Squeeze(tuple(squeeze_axes))
        s = "".join(common)
        s_set = frozenset(s)

    missing = t_set - s_set
    if missing:
        # source has missing dims: expand them
        yield Unsqueeze(tuple(range(len(missing))))
        s = "".join(missing) + s
        s_set = frozenset(s)

    # source has the same dims: transpose them
    assert s_set == t_set
    n = len(s)
    sbuf = bytearray(s.encode())
    tbuf = t.encode()
    while sbuf != tbuf:
        min_distance = np.inf
        for candidate_transpose in _iter_transpositions(n):
            buf = bytearray(sbuf)
            candidate_transpose.transform_sequence(buf)
            distance = levenshtein(buf.decode(), t)
            if distance < min_distance:
                best_transpose = candidate_transpose
                min_distance = distance
        yield best_transpose
        best_transpose.transform_sequence(sbuf)


def _iter_transpositions(n: int) -> Iterator[AxesMapper]:
    for i in range(n):
        for j in range(i + 1, n):
            yield Swap(i, j)
            yield Move(i, j)
            yield Move(j, i)
