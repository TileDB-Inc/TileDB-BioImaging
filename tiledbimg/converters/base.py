import os
from abc import ABC, abstractmethod
from concurrent import futures
from typing import Any, Optional, Sequence

import numpy as np
import tiledb


class ImageConverter(ABC):
    _rgb_dtype = np.dtype([("", "uint8"), ("", "uint8"), ("", "uint8")])

    def __init__(self, max_x_tile: int = 1024, max_y_tile: int = 1024):
        self.max_x_tile = max_x_tile
        self.max_y_tile = max_y_tile

    @abstractmethod
    def convert_image(
        self, input_path: str, output_group_path: str, level_min: int = 0
    ) -> None:
        """
        Convert an image to a TileDB Group of Arrays, one per level.

        :param input_path: path to the input image
        :param output_group_path: path to the TileDB group of arrays
        :param level_min: minimum level of the image to be converted. By default set to 0
            to convert all levels.
        """

    def convert_images(
        self,
        input_paths: Sequence[str],
        output_path: str,
        level_min: int = 0,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Convert a batch of images to TileDB Groups of Arrays (one per level)

        :param input_paths: paths to the input images
        :param output_path: parent directory of the paths to the TiledDB groups of arrays
        :param level_min: minimum level of the image to be converted. By default set to 0
            to convert all levels.
        :param max_workers: Number of parallel workers to convert the images. By default
            (None) all cores are used. Pass 0 for sequential conversion.
        """

        def get_group_path(p: str) -> str:
            return os.path.join(output_path, os.path.splitext(os.path.basename(p))[0])

        if max_workers != 0:
            with futures.ProcessPoolExecutor(max_workers) as executor:
                fs = [
                    executor.submit(
                        self.convert_image,
                        input_path,
                        get_group_path(input_path),
                        level_min,
                    )
                    for input_path in input_paths
                ]
                futures.wait(fs)
                for f in fs:
                    # reraise exception raised on worker
                    f.result()
        else:
            for input_path in input_paths:
                self.convert_image(input_path, get_group_path(input_path), level_min)

    def _write_level(self, output_group_path: str, level: int, data: np.ndarray) -> str:
        assert data.ndim == 3 and data.shape[2] == 3, data.shape
        uri = os.path.join(output_group_path, f"l_{level}.tdb")
        schema = self.__get_schema(*data.shape[:2])
        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, "w") as A:
            A[:] = np.ascontiguousarray(data).view(dtype=self._rgb_dtype)
        return uri

    @staticmethod
    def _write_metadata(
        output_group_path: str,
        input_path: str,
        level_downsamples: Any = None,
        uris: Sequence[str] = (),
    ) -> None:
        with tiledb.Group(output_group_path, "w") as G:
            G.meta["original_filename"] = input_path
            if level_downsamples is not None:
                G.meta["level_downsamples"] = level_downsamples
            for level_uri in uris:
                G.add(os.path.basename(level_uri), relative=True)

    def __get_schema(self, x_size: int, y_size: int) -> tiledb.ArraySchema:
        return tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(
                    name="X",
                    domain=(0, x_size - 1),
                    dtype=np.uint64,
                    tile=min(x_size, self.max_x_tile),
                ),
                tiledb.Dim(
                    name="Y",
                    domain=(0, y_size - 1),
                    dtype=np.uint64,
                    tile=min(y_size, self.max_y_tile),
                ),
            ),
            attrs=[
                tiledb.Attr(
                    name="rgb",
                    dtype=self._rgb_dtype,
                    filters=[tiledb.ZstdFilter(level=0)],
                )
            ],
        )
