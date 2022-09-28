import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import tiledb


@dataclass(frozen=True)
class ImageConverter(object):
    tile_x: int = 1024
    tile_y: int = 1024

    @staticmethod
    def output_level_path(base_path: str, level: int) -> str:
        return os.path.join(base_path, f"l_{level}.tdb")

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
