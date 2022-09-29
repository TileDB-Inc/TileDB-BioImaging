import os
from abc import ABC, abstractmethod
from typing import Any, Sequence

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
        :param output_group_path: path to the TildDB group of arrays
        :param level_min: minimum level of the image to be converted. By default set to 0
            to convert all levels.
        """

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
