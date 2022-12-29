import multiprocessing
import os
from enum import Enum
from functools import partial
from typing import List, Tuple
import psutil
import math

import numpy as np
import tiledb
from tiledb.bioimg.converters.axes import Axes
from scipy.ndimage import zoom

from multiprocessing import Pool

import cv2
import numpy

from tiledb.bioimg.converters.tiles import iter_tiles


class ScaleMethod(Enum):
    NEAREST = 0,
    LINEAR = 1,
    QUADRATIC = 2,
    CUBIC = 3


class Scaler(object):
    def __init__(self, base: tiledb.array.DenseArray, axes: Axes, num_levels: int = -1,
                 scale_factor: int | List[int] = None):
        self._scale_factor = scale_factor
        self._scale_factor_progressive = [scale_factor[0] if isinstance(scale_factor, list) else scale_factor]

        for index in range(1, len(scale_factor)):
            self._scale_factor_progressive.append(scale_factor[index] // scale_factor[index - 1])

        self._initial_shape = base.shape

        self._resolutions = []
        self._zoom_levels = []
        self._zoom_levels_progressive = []

        self._tiles = []
        self._x_index = axes.dims.find("X")
        self._y_index = axes.dims.find("Y")

        if isinstance(scale_factor, list):
            print("Warning: num_levels will be ignored since multiple scale factors where provided")
            for level in range(len(scale_factor)):
                ratio = scale_factor[level]

                shape = list(self._initial_shape)
                zoom_factor = np.ones(len(shape), float)
                zoom_factor_progressive = np.ones(len(shape), float)

                shape[self._x_index] = self._initial_shape[self._x_index] // ratio
                shape[self._y_index] = self._initial_shape[self._y_index] // ratio

                zoom_factor[self._x_index] = zoom_factor[self._y_index] = 1.0 / ratio
                zoom_factor_progressive[self._x_index] = zoom_factor_progressive[self._y_index] = scale_factor[
                                                                                                      level - 1] / ratio if level > 0 else 1.0 / ratio

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
                self._zoom_levels.append(tuple(zoom_factor))
                self._zoom_levels_progressive.append(tuple(zoom_factor_progressive))
                self._tiles.append(tiles)
        else:
            for level in range(1, num_levels + 1):
                shape = list(self._initial_shape)
                zoom_factor = np.ones(len(shape), float)
                zoom_factor_progressive = np.ones(len(shape), float)

                shape[self._x_index] = self._initial_shape[self._x_index] // 2 ** level
                shape[self._y_index] = self._initial_shape[self._y_index] // 2 ** level

                zoom_factor[self._x_index] = zoom_factor[self._y_index] = 1.0 / 2 ** level
                zoom_factor_progressive[self._x_index] = zoom_factor_progressive[self._y_index] = 1.0 / 2

                self._resolutions.append(tuple(shape))
                self._zoom_levels.append(tuple(zoom_factor))
                self._zoom_levels_progressive.append(tuple(zoom_factor_progressive))

    @property
    def resolutions(self) -> List[Tuple[int, ...]]:
        return self._resolutions

    @property
    def tiles(self):
        return self._tiles

    def apply(self, input_data: numpy.ndarray, level: int, method: ScaleMethod) -> numpy.ndarray:
        if input_data.shape != self._initial_shape:
            raise ValueError(f"Incompatible input shape. Expected {self._initial_shape}, got {input_data.shape}")

        if level >= len(self._zoom_levels):
            raise ValueError(f"Unknown zoom level. Expected 0 to {len(self._zoom_levels) - 1}, got {level}")

        result = zoom(input_data, zoom=self._zoom_levels[level], order=method.value, mode="nearest")

        if result.shape != self._resolutions[level]:
            raise ValueError(f"Resizing failed. Expected shape {self._resolutions[level]}, got {result.shape}")

        return result

    def apply_progressive(self, input_data: numpy.ndarray, method: ScaleMethod) -> numpy.ndarray:

        level = 0 if input_data.shape == self._initial_shape else self.resolutions.index(input_data.shape) + 1

        if level >= len(self._zoom_levels):
            raise ValueError(f"Unknown zoom level. Expected 0 to {len(self._zoom_levels) - 1}, got {level}")

        result = zoom(input_data, zoom=self._zoom_levels_progressive[level], order=method.value, mode="nearest")

        if result.shape != self._resolutions[level]:
            raise ValueError(f"Resizing failed. Expected shape {self._resolutions[level]}, got {result.shape}")

        return result

    def apply_chunked(self, base: tiledb.array.DenseArray, out_data: tiledb.array.DenseArray, level: int,
                      method: ScaleMethod) -> None:
        if base.shape != self._initial_shape:
            raise ValueError(f"Incompatible input shape. Expected {self._initial_shape}, got {base.shape}")

        if level >= len(self._zoom_levels):
            raise ValueError(f"Unknown zoom level. Expected 0 to {len(self._zoom_levels) - 1}, got {level}")

        with Pool(multiprocessing.cpu_count()) as pool:
            pool.map(partial(self._scale, base=base, out_data=out_data, level=level, scale_method=method.value[0]),
                     iter_tiles(out_data.domain))

    def apply_chunked_progressive(self, base: tiledb.array.DenseArray, out_data: tiledb.array.DenseArray,
                                  method: ScaleMethod) -> None:
        level = 0 if base.shape == self._initial_shape else self.resolutions.index(base.shape) + 1

        if level >= len(self._zoom_levels):
            raise ValueError(f"Unknown zoom level. Expected 0 to {len(self._zoom_levels) - 1}, got {level}")

        process = psutil.Process(os.getpid())
        available_mem = psutil.virtual_memory()[1] / 2 ** 20
        memory_per_process = (math.prod([dim.tile for dim in base.domain]) * (
                    (1 if method.value[0] < 2 else 2) * self._scale_factor_progressive[level] * 2 + 1) + process.memory_info().rss) / 2 ** 20

        max_num_of_processes = int(min(math.floor(available_mem / memory_per_process), multiprocessing.cpu_count()))

        print(f"Available RAM {available_mem} MB")
        print(f"Estimated RAM requirements per process {memory_per_process} MB")
        print(f"{max_num_of_processes} processes will be spawn for parallel pyramid generation")

        with Pool(max_num_of_processes) as pool:
            pool.map(partial(self._scale_progressive, base=base, out_data=out_data, level=level,
                             scale_method=method.value[0]), iter_tiles(out_data.domain))

    def _scale(self, tile: Tuple[slice], base: tiledb.array.DenseArray, out_data: tiledb.array.DenseArray, level: int,
               scale_method: int) -> None:
        scaled_tile = list(tile)

        scaled_tile[self._x_index] = slice(tile[self._x_index].start * self._scale_factor[level],
                                           min(tile[self._x_index].stop * self._scale_factor[level],
                                               base.shape[self._x_index]))
        scaled_tile[self._y_index] = slice(tile[self._y_index].start * self._scale_factor[level],
                                           min(tile[self._y_index].stop * self._scale_factor[level],
                                               base.shape[self._y_index]))

        out_data[tile] = zoom(input=base[tuple(scaled_tile)], zoom=self._zoom_levels[level], order=scale_method,
                              mode="nearest", grid_mode=True)

    def _scale_progressive(self, tile: Tuple[slice], base: tiledb.array.DenseArray, out_data: tiledb.array.DenseArray,
                           level: int, scale_method: int) -> None:
        scaled_tile = list(tile)

        scaled_tile[self._x_index] = slice(tile[self._x_index].start * self._scale_factor_progressive[level],
                                           min(tile[self._x_index].stop * self._scale_factor_progressive[level],
                                               base.shape[self._x_index]))
        scaled_tile[self._y_index] = slice(tile[self._y_index].start * self._scale_factor_progressive[level],
                                           min(tile[self._y_index].stop * self._scale_factor_progressive[level],
                                               base.shape[self._y_index]))

        out_data[tile] = zoom(input=base[tuple(scaled_tile)], zoom=self._zoom_levels_progressive[level],
                              order=scale_method, mode="nearest", grid_mode=True)
