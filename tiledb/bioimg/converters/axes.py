from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from operator import itemgetter
from typing import Iterator, List, MutableSequence, Sequence, TypeVar

import Levenshtein
import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class Transpose(ABC):
    i: int
    j: int

    def transposed(self, s: Sequence[T]) -> List[T]:
        """Return the transposed version of the given sequence"""
        s = list(s)
        self.transpose(s)
        return s

    @abstractmethod
    def transpose(self, s: MutableSequence[T]) -> None:
        """Transpose the given mutable sequence in place"""

    @abstractmethod
    def transposed_array(self, a: np.ndarray) -> np.ndarray:
        """Return the transposed version of the given numpy array"""


class Swap(Transpose):
    def transpose(self, s: MutableSequence[T]) -> None:
        i, j = self.i, self.j
        s[i], s[j] = s[j], s[i]

    def transposed_array(self, a: np.ndarray) -> np.ndarray:
        return np.swapaxes(a, self.i, self.j)


class Move(Transpose):
    def transpose(self, s: MutableSequence[T]) -> None:
        s.insert(self.j, s.pop(self.i))

    def transposed_array(self, a: np.ndarray) -> np.ndarray:
        return np.moveaxis(a, self.i, self.j)


def minimize_transpositions(s: Sequence[T], t: Sequence[T]) -> Sequence[Transpose]:
    assert Counter(s) == Counter(t)
    n = len(s)
    s = list(s)
    t = list(t)
    transpositions = []
    while s != t:
        weighted_transpositions = (
            (Levenshtein.distance(transposition.transposed(s), t), transposition)
            for transposition in gen_transpositions(n)
        )
        best_transposition = min(weighted_transpositions, key=itemgetter(0))[1]
        best_transposition.transpose(s)
        transpositions.append(best_transposition)
    return transpositions


def gen_transpositions(n: int) -> Iterator[Transpose]:
    for i in range(n):
        for j in range(i + 1, n):
            yield Swap(i, j)
            yield Move(i, j)
            yield Move(j, i)


class Axes:
    _CANONICAL_MEMBERS = "TCZYX"

    def __init__(self, members: str) -> None:
        axes = set(members)
        if len(members) != len(axes):
            raise ValueError(f"Duplicate axes: {members}")
        axes.difference_update(self._CANONICAL_MEMBERS)
        if axes:
            raise ValueError(f"{axes.pop()!r} is not a valid Axis")
        self.members = members

    def canonical(self) -> Axes:
        """Return an Axes instance with the same axis members as this one in canonical order"""
        return Axes("".join(m for m in self._CANONICAL_MEMBERS if m in self.members))

    def transpose(self, a: np.ndarray) -> np.ndarray:
        """Transpose the given array to the canonical axes order"""
        for t in minimize_transpositions(self.members, self.canonical().members):
            a = t.transposed_array(a)
        return a
