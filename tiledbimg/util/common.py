import os
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import tiledb


@dataclass
class LevelTile:
    x: int
    y: int


class ImageConverter(object):
    input_image: str
    output_group: str

    tile_x: int = 1024
    tile_y: int = 1024

    @staticmethod
    def output_level_path(base_path: str, level: int) -> str:
        return os.path.join(base_path, f"l_{level}.tdb")

    def level_tile(self, shape: Tuple[Any, ...]) -> LevelTile:
        level_tile = LevelTile(
            np.min((self.tile_x, shape[0])), np.min((self.tile_y, shape[1]))
        )
        return level_tile

    def create_schema(self, img_shape: Tuple[Any, ...]) -> tiledb.ArraySchema:
        img_shape = tuple((img_shape[0], img_shape[1], 3))  # swappity
        print("Processing level with shape: ", img_shape)
        dims = []

        level_tile = self.level_tile(img_shape)

        dims.append(
            tiledb.Dim(
                name="X",
                domain=(0, img_shape[0] - 1),
                dtype=np.uint64,
                tile=level_tile.x,
            )
        )
        dims.append(
            tiledb.Dim(
                name="Y",
                domain=(0, img_shape[1] - 1),
                dtype=np.uint64,
                tile=level_tile.y,
            )
        )

        filters = [tiledb.ZstdFilter(level=0)]
        attr = tiledb.Attr(
            name="rgb",
            dtype=[("", "uint8"), ("", "uint8"), ("", "uint8")],
            filters=filters,
        )

        schema = tiledb.ArraySchema(tiledb.Domain(*dims), attrs=[attr])
        return schema
