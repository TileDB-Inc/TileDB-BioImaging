from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, MutableSequence, Tuple, TypeVar

import numpy as np
from pyeditdistance.distance import levenshtein

T = TypeVar("T")


class Transform(ABC):
    @abstractmethod
    def __call__(self, s: MutableSequence[T], **kwargs: Any) -> None:
        """Transform the given mutable sequence in place"""

    @abstractmethod
    def map_array(self, a: np.ndarray) -> np.ndarray:
        """Return the transformed numpy array"""


@dataclass(frozen=True)
class Swap(Transform):
    i: int
    j: int

    def __call__(self, s: MutableSequence[T], **kwargs: Any) -> None:
        i, j = self.i, self.j
        s[i], s[j] = s[j], s[i]

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return np.swapaxes(a, self.i, self.j)


@dataclass(frozen=True)
class Move(Transform):
    i: int
    j: int

    def __call__(self, s: MutableSequence[T], **kwargs: Any) -> None:
        s.insert(self.j, s.pop(self.i))

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return np.moveaxis(a, self.i, self.j)


@dataclass(frozen=True)
class Squeeze(Transform):
    idxs: Tuple[int, ...]

    def __call__(self, s: MutableSequence[T], **kwargs: Any) -> None:
        for i in sorted(self.idxs, reverse=True):
            del s[i]

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return np.squeeze(a, self.idxs)


@dataclass(frozen=True)
class Unsqueeze(Transform):
    idxs: Tuple[int, ...]

    def __call__(self, s: MutableSequence[T], **kwargs: Any) -> None:
        fill_value = kwargs["fill_value"]
        for i in sorted(self.idxs):
            s.insert(i, fill_value)

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return np.expand_dims(a, self.idxs)


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


class AxesMapper:
    def __init__(self, source: Axes, target: Axes):
        self._transforms = tuple(_iter_transforms(source.dims, target.dims))

    def map_array(self, a: np.ndarray) -> np.ndarray:
        """Transform a Numpy array from the source axes `s` to the target axes `t`.

        If `s` is a superset of `t`, squeeze the extra axes.
        If `s` is a subset of `t`, insert the missing axes at the front with length one.
        Finally, find the minimum number of transforms from `s` to `t` and apply them to `a`.
        """
        for transform in self._transforms:
            a = transform.map_array(a)
        return a

    def map_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Transform the shape of a Numpy array from the source to the target axes."""
        mapped_shape = list(shape)
        for transform in self._transforms:
            transform(mapped_shape, fill_value=1)
        return tuple(mapped_shape)


def _iter_transforms(s: str, t: str) -> Iterator[Transform]:
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
            candidate_transpose(buf)
            distance = levenshtein(buf.decode(), t)
            if distance < min_distance:
                best_transpose = candidate_transpose
                min_distance = distance
        yield best_transpose
        best_transpose(sbuf)


def _iter_transpositions(n: int) -> Iterator[Transform]:
    for i in range(n):
        for j in range(i + 1, n):
            yield Swap(i, j)
            yield Move(i, j)
            yield Move(j, i)
