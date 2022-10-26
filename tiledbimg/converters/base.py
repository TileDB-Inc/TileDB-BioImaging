import os
from abc import ABC, abstractmethod
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlparse

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

    def level_metadata(self, level: int) -> Dict[str, Any]:
        return {}

    def metadata(self) -> Dict[str, Any]:
        return {}


@dataclass(frozen=True)
class Dimension:
    name: str
    max_tile: int

    def to_tiledb_dim(self, size: int, dtype: np.dtype) -> tiledb.Dim:
        return tiledb.Dim(
            name=self.name,
            domain=(0, size - 1),
            dtype=dtype,
            tile=min(size, self.max_tile),
        )


class ImageConverter(ABC):
    def __init__(
        self,
        c_dim: Dimension = Dimension("C", 3),  # channel
        y_dim: Dimension = Dimension("Y", 1024),  # height
        x_dim: Dimension = Dimension("X", 1024),  # width
    ):
        self._dims = (c_dim, y_dim, x_dim)

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
            uri = os.path.join(output_group_path, f"l_{level}.tdb")
            image = reader.level_image(level)
            level_metadata = reader.level_metadata(level)
            level_metadata["level"] = level
            self._write_image(uri, image, level_metadata)
            uris.append(uri)

        # Write group metadata
        with tiledb.Group(output_group_path, "w") as G:
            metadata = reader.metadata()
            if metadata:
                G.meta.update(metadata)
            for level_uri in uris:
                if urlparse(level_uri).scheme == "tiledb":
                    G.add(level_uri, relative=False)
                else:
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

    def _write_image(
        self, uri: str, image: np.ndarray, metadata: Dict[str, Any]
    ) -> None:
        assert len(image.shape) == len(self._dims)
        # find the smallest dtype that can hold the number of image scalar values
        dim_dtype = np.min_scalar_type(image.size)
        dims = (
            dim.to_tiledb_dim(size, dim_dtype)
            for dim, size in zip(self._dims, image.shape)
        )
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(*dims),
            attrs=[
                tiledb.Attr(
                    name="",
                    dtype=image.dtype,
                    filters=[tiledb.ZstdFilter(level=0)],
                )
            ],
        )
        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, "w") as A:
            A[:] = image
            if metadata:
                A.meta.update(metadata)
