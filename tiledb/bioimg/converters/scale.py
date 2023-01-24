import math
import multiprocessing
from enum import Enum
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import psutil
from skimage.transform import resize

import tiledb

from .axes import Axes
from .tiles import iter_tiles


class ScaleMethod(Enum):
    NEAREST = 0
    LINEAR = 1


class ScalerMode(Enum):
    NON_PROGRESSIVE = 0
    PROGRESSIVE = 1
    CHUNKED_NON_PROGRESSIVE = 2
    CHUNKED_PROGRESSIVE = 3


class Scaler(object):
    def __init__(
        self, base: tiledb.array.DenseArray, axes: Axes, scale_factor: List[int]
    ):
        self._scale_factor = scale_factor
        self._scale_factor_progressive = [float(scale_factor[0])]

        for index in range(1, len(scale_factor)):
            self._scale_factor_progressive.append(
                scale_factor[index] / float(scale_factor[index - 1])
            )

        self._initial_shape = base.shape

        self._resolutions = []

        self._tiles = []
        self._x_index = axes.dims.find("X")
        self._y_index = axes.dims.find("Y")

        for level in range(len(scale_factor)):
            ratio = scale_factor[level]

            shape = list(self._initial_shape)

            shape[self._x_index] = int(
                round(self._initial_shape[self._x_index] / float(ratio))
            )
            shape[self._y_index] = int(
                round(self._initial_shape[self._y_index] / float(ratio))
            )

            tiles = []

            for tile in iter_tiles(base.domain):
                scaled_tile = list(tile)

                scaled_x_start = tile[self._x_index].start // ratio
                scaled_x_end = tile[self._x_index].stop // ratio
                scaled_tile[self._x_index] = slice(scaled_x_start, scaled_x_end)

                scaled_y_start = tile[self._y_index].start // ratio
                scaled_y_end = tile[self._y_index].stop // ratio
                scaled_tile[self._y_index] = slice(scaled_y_start, scaled_y_end)

                tiles.append(tuple(tile))

            self._resolutions.append(tuple(shape))
            self._tiles.append(tiles)

    @property
    def resolutions(self) -> List[Tuple[int, ...]]:
        return self._resolutions

    @property
    def tiles(self) -> List[List[Tuple[slice, ...]]]:
        return self._tiles

    def apply(
        self,
        input_data: tiledb.array.DenseArray,
        out_data: tiledb.array.DenseArray,
        level: int,
        method: ScaleMethod,
    ) -> None:
        if input_data.shape != self._initial_shape:
            raise ValueError(
                f"Incompatible input shape. Expected {self._initial_shape}, got {input_data.shape}"
            )

        if level >= len(self._resolutions):
            raise ValueError(
                f"Unknown zoom level. Expected 0 to {len(self._resolutions) - 1}, got {level}"
            )

        out_data[:] = resize(
            input_data[:],
            self._resolutions[level],
            preserve_range=True,
            order=method.value,
            anti_aliasing=True,
        )

    def apply_progressive(
        self,
        input_data: tiledb.array.DenseArray,
        out_data: tiledb.array.DenseArray,
        method: ScaleMethod,
    ) -> None:

        level = (
            0
            if input_data.shape == self._initial_shape
            else self.resolutions.index(input_data.shape) + 1
        )

        if level >= len(self._resolutions):
            raise ValueError(
                f"Unknown zoom level. Expected 0 to {len(self._resolutions) - 1}, got {level}"
            )

        out_data[:] = resize(
            input_data[:],
            self._resolutions[level],
            preserve_range=True,
            order=method.value,
            anti_aliasing=True,
        )

    def apply_chunked(
        self,
        base: tiledb.array.DenseArray,
        out_data: tiledb.array.DenseArray,
        level: int,
        method: ScaleMethod,
    ) -> None:
        if base.shape != self._initial_shape:
            raise ValueError(
                f"Incompatible input shape. Expected {self._initial_shape}, got {base.shape}"
            )

        if level >= len(self._resolutions):
            raise ValueError(
                f"Unknown zoom level. Expected 0 to {len(self._resolutions) - 1}, got {level}"
            )

        available_mem = psutil.virtual_memory()[1] / 2**20
        memory_per_process = (
            math.prod([dim.tile for dim in base.domain])
            * ((1 if method.value < 2 else 2) * self._scale_factor[level] * 2 + 1)
            + 120 * 2**20
        ) / 2**20

        max_num_of_processes = int(
            min(
                math.floor(available_mem / memory_per_process),
                multiprocessing.cpu_count(),
            )
        )

        with Pool(max_num_of_processes) as pool:
            pool.map(
                partial(
                    self._scale,
                    base=base,
                    out_data=out_data,
                    level=level,
                    scale_method=method.value,
                ),
                iter_tiles(out_data.domain),
            )

    def apply_chunked_progressive(
        self,
        base: tiledb.array.DenseArray,
        out_data: tiledb.array.DenseArray,
        method: ScaleMethod,
    ) -> None:
        level = (
            0
            if base.shape == self._initial_shape
            else self.resolutions.index(base.shape) + 1
        )

        if level >= len(self._resolutions):
            raise ValueError(
                f"Unknown zoom level. Expected 0 to {len(self._resolutions) - 1}, got {level}"
            )

        available_mem = psutil.virtual_memory()[1] / 2**20
        memory_per_process = (
            math.prod([dim.tile for dim in base.domain])
            * (
                (1 if method.value < 2 else 2)
                * self._scale_factor_progressive[level]
                * 2
                + 1
            )
            + 120 * 2**20
        ) / 2**20

        max_num_of_processes = int(
            min(
                math.floor(available_mem / memory_per_process),
                multiprocessing.cpu_count(),
            )
        )

        with Pool(max_num_of_processes) as pool:
            pool.map(
                partial(
                    self._scale_progressive,
                    base=base,
                    out_data=out_data,
                    level=level,
                    scale_method=method.value,
                ),
                iter_tiles(out_data.domain),
            )

    def _scale(
        self,
        tile: Tuple[slice],
        base: tiledb.array.DenseArray,
        out_data: tiledb.array.DenseArray,
        level: int,
        scale_method: int,
    ) -> None:
        scaled_tile = list(tile)
        slice_shape = []
        for sl in scaled_tile:
            slice_shape.append(sl.stop - sl.start)

        scaled_tile[self._x_index] = slice(
            int(tile[self._x_index].start * self._scale_factor[level]),
            int(
                min(
                    tile[self._x_index].stop * self._scale_factor[level],
                    base.shape[self._x_index],
                )
            ),
        )
        scaled_tile[self._y_index] = slice(
            int(tile[self._y_index].start * self._scale_factor[level]),
            int(
                min(
                    tile[self._y_index].stop * self._scale_factor[level],
                    base.shape[self._y_index],
                )
            ),
        )

        out_data[tile] = resize(
            base[tuple(scaled_tile)],
            tuple(slice_shape),
            preserve_range=True,
            order=scale_method,
            anti_aliasing=True,
        )

    def _scale_progressive(
        self,
        tile: Tuple[slice],
        base: tiledb.array.DenseArray,
        out_data: tiledb.array.DenseArray,
        level: int,
        scale_method: int,
    ) -> None:
        scaled_tile = list(tile)
        slice_shape = []
        for sl in scaled_tile:
            slice_shape.append(sl.stop - sl.start)

        scaled_tile[self._x_index] = slice(
            int(tile[self._x_index].start * self._scale_factor_progressive[level]),
            int(
                min(
                    tile[self._x_index].stop * self._scale_factor_progressive[level],
                    base.shape[self._x_index],
                )
            ),
        )
        scaled_tile[self._y_index] = slice(
            int(tile[self._y_index].start * self._scale_factor_progressive[level]),
            int(
                min(
                    tile[self._y_index].stop * self._scale_factor_progressive[level],
                    base.shape[self._y_index],
                )
            ),
        )

        out_data[tile] = resize(
            base[tuple(scaled_tile)],
            tuple(slice_shape),
            preserve_range=True,
            order=scale_method,
            anti_aliasing=True,
        )
