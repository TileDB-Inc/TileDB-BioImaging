from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, MutableSequence, Sequence, Tuple, TypeVar

import numpy as np
from pyeditdistance.distance import levenshtein

T = TypeVar("T")


class Transform(ABC):
    @abstractmethod
    def transform(self, s: MutableSequence[T]) -> None:
        """Transform the given mutable sequence in place"""

    @abstractmethod
    def transformed_array(self, a: np.ndarray) -> np.ndarray:
        """Return the transformed numpy array"""


@dataclass(frozen=True)
class Swap(Transform):
    i: int
    j: int

    def transform(self, s: MutableSequence[T]) -> None:
        i, j = self.i, self.j
        s[i], s[j] = s[j], s[i]

    def transformed_array(self, a: np.ndarray) -> np.ndarray:
        return np.swapaxes(a, self.i, self.j)


@dataclass(frozen=True)
class Move(Transform):
    i: int
    j: int

    def transform(self, s: MutableSequence[T]) -> None:
        s.insert(self.j, s.pop(self.i))

    def transformed_array(self, a: np.ndarray) -> np.ndarray:
        return np.moveaxis(a, self.i, self.j)


@dataclass(frozen=True)
class Squeeze(Transform):
    idxs: Tuple[int, ...]

    def transform(self, s: MutableSequence[T]) -> None:
        for i in sorted(self.idxs, reverse=True):
            del s[i]

    def transformed_array(self, a: np.ndarray) -> np.ndarray:
        return np.squeeze(a, self.idxs)


@dataclass(frozen=True)
class Unsqueeze(Transform):
    idxs: Tuple[int, ...]
    fill_value: Any

    def transform(self, s: MutableSequence[T]) -> None:
        for i in sorted(self.idxs):
            s.insert(i, self.fill_value)

    def transformed_array(self, a: np.ndarray) -> np.ndarray:
        return np.expand_dims(a, self.idxs)


def transform_array(a: np.ndarray, s: str, t: str) -> np.ndarray:
    """Transform a Numpy array `a` from source `s` axes to target `t`.

    If `s` is a superset of `t`, squeeze the extra axes (provided they are of length one).
    If `s` is a subset of `t`, insert the missing axes at the front with length one.
    Finally, find the minimum number of transforms from `s` to `t` and apply them to `a`.

    :param a: Source array to transform
    :param s: Axes of the source array `a`
    :param t: Axes of the target array
    :return: The transformed array
    """
    assert len(s) == len(a.shape)
    s_set, t_set = frozenset(s), frozenset(t)
    transforms: MutableSequence[Transform] = []

    if s_set > t_set:
        # source has extra dims: squeeze them (assuming their size is 1)
        common, squeeze_axes = [], []
        for i, m in enumerate(s):
            if m in t_set:
                common.append(m)
            else:
                squeeze_axes.append(i)
        s = "".join(common)
        transforms.append(Squeeze(tuple(squeeze_axes)))

    elif s_set < t_set:
        # source has missing dims: expand them
        missing = t_set - s_set
        s = "".join(missing) + s
        transforms.append(Unsqueeze(tuple(range(len(missing))), fill_value=1))

    transforms.extend(minimize_transforms(s, t))
    for transform in transforms:
        a = transform.transformed_array(a)

    return a


def minimize_transforms(s: str, t: str) -> Sequence[Transform]:
    assert Counter(s) == Counter(t)
    n = len(s)
    sbuf = bytearray(s.encode())
    tbuf = t.encode()
    transforms = []
    while sbuf != tbuf:
        min_distance = np.inf
        for transform in gen_transpositions(n):
            buf = bytearray(sbuf)
            transform.transform(buf)
            distance = levenshtein(buf.decode(), t)
            if distance < min_distance:
                best_transform = transform
                min_distance = distance
        best_transform.transform(sbuf)
        transforms.append(best_transform)
    return transforms


def gen_transpositions(n: int) -> Iterator[Transform]:
    for i in range(n):
        for j in range(i + 1, n):
            yield Swap(i, j)
            yield Move(i, j)
            yield Move(j, i)


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
