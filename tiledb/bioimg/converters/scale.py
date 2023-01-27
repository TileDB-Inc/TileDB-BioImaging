import multiprocessing
from enum import IntFlag, auto
from functools import partial
from multiprocessing import Pool
from typing import Sequence, Tuple

from skimage.transform import resize

import tiledb

from .axes import Axes
from .tiles import iter_tiles


class ScalerMode(IntFlag):
    NON_PROGRESSIVE = auto()
    PROGRESSIVE = auto()
    CHUNKED_NON_PROGRESSIVE = auto()
    CHUNKED_PROGRESSIVE = auto()


def _scale_tile(
    tile: Tuple[slice],
    base: tiledb.Array,
    output: tiledb.Array,
    scale_factors: Sequence[float],
    order: int,
) -> None:
    slice_shape = tuple(dimension.stop - dimension.start for dimension in list(tile))

    scaled_tile = []
    for index, dimension in enumerate(tile):
        start = int(dimension.start * scale_factors[index])
        stop = int(min(dimension.stop * scale_factors[index], base.shape[index]))

        scaled_tile.append(slice(start, stop))

    output[tile] = resize(
        base[tuple(scaled_tile)],
        slice_shape,
        preserve_range=True,
        order=order,
        anti_aliasing=True,
    )


class Scaler(object):
    def __init__(
        self,
        base: tiledb.array,
        base_axes: Axes,
        scale_factors: Sequence[float],
        scale_axes: str,
        mode: ScalerMode,
        order: int,
    ):
        self._mode = mode
        self._order = order

        self._scale_factors = []
        self._scale_factors_progressive = []
        self._resolutions = []

        previous_scale_factor = 1.0

        for factor in scale_factors:
            level_factors = [
                factor if axis in scale_axes else 1 for axis in base_axes.dims
            ]
            level_factors_progressive = [
                factor / previous_scale_factor if axis in scale_axes else 1
                for axis in base_axes.dims
            ]
            resolution = tuple(
                round(base.shape[i] / level_factors[i]) for i in range(len(base.shape))
            )

            previous_scale_factor = factor

            self._scale_factors.append(level_factors)
            self._scale_factors_progressive.append(level_factors_progressive)
            self._resolutions.append(resolution)

    @property
    def resolutions(self) -> Sequence[Tuple[int, ...]]:
        return self._resolutions

    def apply(self, base: tiledb.Array, output: tiledb.Array, level: int) -> None:
        if bool(self._mode & (ScalerMode.NON_PROGRESSIVE | ScalerMode.PROGRESSIVE)):
            output[:] = resize(
                base[:],
                output.shape,
                preserve_range=True,
                order=self._order,
                anti_aliasing=True,
            )
        else:
            with Pool(multiprocessing.cpu_count()) as pool:
                scale_factors = (
                    self._scale_factors[level]
                    if self._mode == ScalerMode.CHUNKED_NON_PROGRESSIVE
                    else self._scale_factors_progressive[level]
                )
                pool.map(
                    partial(
                        _scale_tile,
                        base=base,
                        output=output,
                        scale_factors=scale_factors,
                        order=self._order,
                    ),
                    iter_tiles(output.domain),
                )
