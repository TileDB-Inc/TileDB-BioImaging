import multiprocessing
from functools import partial
from multiprocessing import Pool
from typing import Any, Sequence, Tuple

from skimage.transform import resize

import tiledb

from .tiles import iter_tiles


def _scale_tile(
    tile: Tuple[slice, ...],
    base: tiledb.Array,
    output: tiledb.Array,
    scale_factors: Sequence[float],
    **resize_kwargs: Any,
) -> None:
    slice_shape = tuple(dimension.stop - dimension.start for dimension in tile)

    scaled_tile = []
    for index, dimension in enumerate(tile):
        start = int(dimension.start * scale_factors[index])
        stop = int(min(dimension.stop * scale_factors[index], base.shape[index]))

        scaled_tile.append(slice(start, stop))

    output[tile] = resize(base[tuple(scaled_tile)], slice_shape, **resize_kwargs)


class Scaler(object):
    def __init__(
        self,
        base_shape: Tuple[int, ...],
        base_axes: str,
        scale_factors: Sequence[float],
        scale_axes: str,
        chunked: bool = False,
        progressive: bool = False,
        order: int = 1,
    ):
        self._chunked = chunked
        self._order = order

        self._scale_factors = []
        self._level_shapes = []

        previous_scale_factor = 1.0

        for factor in scale_factors:
            dim_factors = [factor if axis in scale_axes else 1 for axis in base_axes]
            self._level_shapes.append(
                tuple(
                    round(dim / dim_factor)
                    for dim, dim_factor in zip(base_shape, dim_factors)
                )
            )
            if chunked:
                if progressive:
                    dim_factors = [
                        factor / previous_scale_factor if axis in scale_axes else 1
                        for axis in base_axes
                    ]
                    previous_scale_factor = factor

                self._scale_factors.append(dim_factors)

    @property
    def level_shapes(self) -> Sequence[Tuple[int, ...]]:
        return self._level_shapes

    def apply(self, base: tiledb.Array, output: tiledb.Array, level: int) -> None:
        if not self._chunked:
            output[:] = resize(
                base[:],
                output.shape,
                preserve_range=True,
                order=self._order,
                anti_aliasing=True,
            )
        else:
            with Pool(multiprocessing.cpu_count()) as pool:
                pool.map(
                    partial(
                        _scale_tile,
                        base=base,
                        output=output,
                        scale_factors=self._scale_factors[level],
                        preserve_range=True,
                        order=self._order,
                        anti_aliasing=True,
                    ),
                    iter_tiles(output.domain),
                )
