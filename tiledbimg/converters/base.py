import os
from abc import ABC, abstractmethod
from concurrent import futures
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import tiledb


class ImageReader(ABC):
    @property
    @abstractmethod
    def level_count(self) -> int:
        """Return the number of levels for this multi-resolution image"""

    @abstractmethod
    def level_image(self, level: int) -> np.ndarray:
        """Return the image for the given level as (X, Y, C) 3D numpy array"""


@dataclass(frozen=True)
class Dimension:
    name: str
    max_tile: int

    def to_tiledb_dim(self, size: int) -> tiledb.Dim:
        return tiledb.Dim(
            name=self.name,
            domain=(0, size - 1),
            dtype=np.uint64,
            tile=min(size, self.max_tile),
        )


class ImageConverter(ABC):
    def __init__(
        self,
        x_dim: Dimension = Dimension("X", 1024),
        y_dim: Dimension = Dimension("Y", 1024),
        c_dim: Dimension = Dimension("C", 3),
    ):
        self._dims = (x_dim, y_dim, c_dim)

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
        tiledb.group_create(output_group_path)
        reader = self._get_image_reader(input_path)

        # Create a TileDB array for each level in range(level_min, reader.level_count)
        uris = []
        for level in range(level_min, reader.level_count):
            image = reader.level_image(level)
            assert image.ndim == 3, image.shape
            uri = os.path.join(output_group_path, f"l_{level}.tdb")
            schema = self.__get_schema(image.shape)
            tiledb.Array.create(uri, schema)
            with tiledb.open(uri, "w") as A:
                A[:] = image
            uris.append(uri)

        # Write metadata
        with tiledb.Group(output_group_path, "w") as G:
            G.meta["original_filename"] = input_path
            for level_uri in uris:
                G.add(os.path.basename(level_uri), relative=True)

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

    @abstractmethod
    def _get_image_reader(self, input_path: str) -> ImageReader:
        """Return an ImageReader for the given input path."""

    def __get_schema(self, shape: Sequence[int]) -> tiledb.ArraySchema:
        assert len(shape) == len(self._dims)
        return tiledb.ArraySchema(
            domain=tiledb.Domain(
                *(dim.to_tiledb_dim(size) for dim, size in zip(self._dims, shape))
            ),
            attrs=[
                tiledb.Attr(
                    name="",
                    dtype=np.uint8,
                    filters=[tiledb.ZstdFilter(level=0)],
                )
            ],
        )
