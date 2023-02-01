from concurrent import futures
from typing import Any, Optional, Sequence, Tuple

import skimage as sk

import tiledb

from .tiles import iter_tiles


class Scaler(object):
    def __init__(
        self,
        base_shape: Tuple[int, ...],
        base_axes: str,
        scale_factors: Sequence[float],
        scale_axes: str = "XY",
        chunked: bool = False,
        progressive: bool = False,
        order: int = 1,
        max_workers: Optional[int] = None,
    ):
        self._chunked = chunked
        self._progressive = progressive
        self._resize_kwargs = dict(order=order, preserve_range=True, anti_aliasing=True)
        self._executor = (
            futures.ProcessPoolExecutor(max_workers) if max_workers != 0 else None
        )
        self._level_shapes = []
        self._scale_factors = []

        previous_scale_factor = 1.0
        for scale_factor in scale_factors:
            dim_factors = [
                scale_factor if axis in scale_axes else 1 for axis in base_axes
            ]
            self._level_shapes.append(
                tuple(
                    round(dim_size / dim_factor)
                    for dim_size, dim_factor in zip(base_shape, dim_factors)
                )
            )
            if chunked:
                if progressive:
                    dim_factors = [
                        scale_factor / previous_scale_factor
                        if axis in scale_axes
                        else 1
                        for axis in base_axes
                    ]
                    previous_scale_factor = scale_factor
                self._scale_factors.append(dim_factors)

    @property
    def level_shapes(self) -> Sequence[Tuple[int, ...]]:
        return self._level_shapes

    @property
    def chunked(self) -> bool:
        return self._chunked

    @property
    def progressive(self) -> bool:
        return self._progressive

    def apply(
        self, in_array: tiledb.Array, out_array: tiledb.Array, level: int
    ) -> None:
        scale_kwargs = dict(
            in_array=in_array,
            out_array=out_array,
            scale_factors=self._scale_factors[level] if self._scale_factors else None,
            **self._resize_kwargs,
        )

        if not self._chunked:
            _scale(**scale_kwargs)
        elif self._executor:
            fs = [
                self._executor.submit(_scale, tile=tile, **scale_kwargs)
                for tile in iter_tiles(out_array.domain)
            ]
            futures.wait(fs)
        else:
            for tile in iter_tiles(out_array.domain):
                _scale(tile=tile, **scale_kwargs)


def _scale(
    in_array: tiledb.Array,
    out_array: tiledb.Array,
    tile: Optional[Tuple[slice, ...]] = None,
    scale_factors: Sequence[float] = (),
    **resize_kwargs: Any,
) -> None:
    if tile is None:
        tile = tuple(slice(0, size) for size in out_array.shape)
        image = in_array[:]
    else:
        scaled_tile = []
        in_shape = in_array.shape
        assert len(tile) == len(scale_factors) == len(in_shape)
        for tile_slice, scale_factor, dim_size in zip(tile, scale_factors, in_shape):
            start = int(tile_slice.start * scale_factor)
            stop = int(min(tile_slice.stop * scale_factor, dim_size))
            scaled_tile.append(slice(start, stop))
        image = in_array[tuple(scaled_tile)]

    tile_shape = tuple(s.stop - s.start for s in tile)
    out_array[tile] = sk.transform.resize(image, tile_shape, **resize_kwargs)
