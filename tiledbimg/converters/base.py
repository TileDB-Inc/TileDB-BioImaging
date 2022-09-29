import os
from abc import ABC, abstractmethod
from concurrent import futures
from typing import Any, Optional, Sequence

import numpy as np
import tiledb


class ImageConverter(ABC):
    def __init__(self, tile_x: int = 1024, tile_y: int = 1024):
        self.tile_x = tile_x
        self.tile_y = tile_y

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

    def create_schema(self, img_shape: Sequence[Any]) -> tiledb.ArraySchema:
        # FIXME: The next line is either redundant or wrong
        img_shape = tuple((img_shape[0], img_shape[1], 3))  # swappity
        return tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(
                    name="X",
                    domain=(0, img_shape[0] - 1),
                    dtype=np.uint64,
                    tile=min(self.tile_x, img_shape[0]),
                ),
                tiledb.Dim(
                    name="Y",
                    domain=(0, img_shape[1] - 1),
                    dtype=np.uint64,
                    tile=min(self.tile_y, img_shape[1]),
                ),
            ),
            attrs=[
                tiledb.Attr(
                    name="rgb",
                    dtype=[("", "uint8"), ("", "uint8"), ("", "uint8")],
                    filters=[tiledb.ZstdFilter(level=0)],
                )
            ],
        )

    @staticmethod
    def output_level_path(base_path: str, level: int) -> str:
        return os.path.join(base_path, f"l_{level}.tdb")
