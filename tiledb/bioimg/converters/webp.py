from typing import Tuple, TypeVar

import numpy as np

from .axes import Axes, AxesMapper


class ToWebPAxesMapper(AxesMapper):
    """Mapper from 3D axes (YXC or permutations thereof) to 2D (YX)"""

    def __init__(self, source: Axes, c_size: int):
        super().__init__(source, target=Axes("YXC"))
        self._source = source
        self._c_size = c_size

    @property
    def inverse(self) -> AxesMapper:
        return FromWebPAxesMapper(self._source, self._c_size)

    def map_array(self, a: np.ndarray) -> np.ndarray:
        s = self.map_shape(a.shape)
        return super().map_array(a).reshape(s)

    def map_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        y, x, c = super().map_shape(shape)
        if c != self._c_size:
            raise ValueError(
                f"{self._c_size} channels expected, {c} given in input_shape: {shape}"
            )
        return y, x * c

    def map_tile(self, tile: Tuple[slice, ...]) -> Tuple[slice, ...]:
        y, x, c = super().map_tile(tile)
        if c != slice(None):
            raise ValueError(f"Only full slice for C dimension is supported, {c} given")
        xc = slice(x.start * self._c_size, x.stop * self._c_size)
        return y, xc


Idx = TypeVar("Idx", int, slice)


class FromWebPAxesMapper(AxesMapper):
    """Mapper from 2D axes (YX) to 3D (YXC or permutations thereof)"""

    def __init__(self, target: Axes, c_size: int):
        super().__init__(Axes("YXC"), target)
        self._target = target
        self._c_size = c_size

    @property
    def inverse(self) -> AxesMapper:
        return ToWebPAxesMapper(self._target, self._c_size)

    def map_array(self, a: np.ndarray) -> np.ndarray:
        return super().map_array(a.reshape(self._to_yxc(a.shape)))

    def map_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return super().map_shape(self._to_yxc(shape))

    def map_tile(self, tile: Tuple[slice, ...]) -> Tuple[slice, ...]:
        return super().map_tile(self._to_yxc(tile))

    def _to_yxc(self, shape_or_tile: Tuple[Idx, ...]) -> Tuple[Idx, ...]:
        c = self._c_size
        y, xc = shape_or_tile
        if isinstance(xc, int):
            assert not isinstance(y, slice)  # assertion added to appease mypy
            return y, xc // c, c
        else:
            return y, slice(xc.start // c, xc.stop // c), slice(None)
