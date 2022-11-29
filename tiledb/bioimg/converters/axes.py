from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from operator import itemgetter
from typing import Iterable, Iterator, Sequence

import numpy as np
from edit_operation import levenshtein


@dataclass(frozen=True)
class Transpose(ABC):
    i: int
    j: int

    def transposed(self, s: bytes) -> bytes:
        """Return the transposed version of the given bytestring"""
        b = bytearray(s)
        self.transpose(b)
        return bytes(b)

    @abstractmethod
    def transpose(self, s: bytearray) -> None:
        """Transpose the given bytearray in place"""

    @abstractmethod
    def transposed_array(self, a: np.ndarray) -> np.ndarray:
        """Return the transposed version of the given numpy array"""


class Swap(Transpose):
    def transpose(self, s: bytearray) -> None:
        i, j = self.i, self.j
        s[i], s[j] = s[j], s[i]

    def transposed_array(self, a: np.ndarray) -> np.ndarray:
        return np.swapaxes(a, self.i, self.j)


class Move(Transpose):
    def transpose(self, s: bytearray) -> None:
        s.insert(self.j, s.pop(self.i))

    def transposed_array(self, a: np.ndarray) -> np.ndarray:
        return np.moveaxis(a, self.i, self.j)


def transpose_array(a: np.ndarray, s: str, t: str) -> np.ndarray:
    """Transpose a Numpy array `a` from source `s` axes to target `t`.

    If `s` is a superset of `t`, squeeze the extra axes (provided they are of length one).
    If `s` is a subset of `t`, insert the missing axes at the front with length one.
    Finally find the minimum number of transpositions from `s` to `t` and apply them to `a`.

    :param a: Source array to transpose
    :param s: Axes of the source array `a`
    :param t: Axes of the target array
    :return: The transposed array
    """
    assert len(s) == len(a.shape)
    s_set, t_set = frozenset(s), frozenset(t)

    if s_set > t_set:
        # source has extra dims: squeeze them (assuming their size is 1)
        common, squeeze_axes = [], []
        for i, m in enumerate(s):
            if m in t_set:
                common.append(m)
            else:
                squeeze_axes.append(i)
        s = "".join(common)
        a = np.squeeze(a, tuple(squeeze_axes))

    elif s_set < t_set:
        # source has missing dims: expand them
        missing = t_set - s_set
        s = "".join(missing) + s
        a = np.expand_dims(a, tuple(range(len(missing))))

    for transposition in minimize_transpositions(s, t):
        a = transposition.transposed_array(a)

    return a


def minimize_transpositions(s: str, t: str) -> Sequence[Transpose]:
    assert Counter(s) == Counter(t)
    n = len(s)
    sbuf = bytearray(s.encode())
    tbuf = t.encode()
    transpositions = []
    while sbuf != tbuf:
        weighted_transpositions = (
            (levenshtein.distance(tr.transposed(sbuf).decode(), t), tr)
            for tr in gen_transpositions(n)
        )
        best_transposition = min(weighted_transpositions, key=itemgetter(0))[1]
        best_transposition.transpose(sbuf)
        transpositions.append(best_transposition)
    return transpositions


def gen_transpositions(n: int) -> Iterator[Transpose]:
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

    def canonical(self, a: np.ndarray) -> Axes:
        """
        Return a new Axes instance with the dimensions of this axes whose size in `a` are
        greater than 1 and ordered in canonical order (TCZYX)
        """
        assert len(self.dims) == len(a.shape)
        dims = frozenset(dim for dim, size in zip(self.dims, a.shape) if size > 1)
        return Axes(dim for dim in self.CANONICAL_DIMS if dim in dims)
