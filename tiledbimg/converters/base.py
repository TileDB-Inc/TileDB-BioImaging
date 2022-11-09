from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Dict
from urllib.parse import urlparse

import numpy as np
import tiledb

from .axes import Axes


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


class ImageReader(ABC):
    def __enter__(self) -> ImageReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Return the number of levels for this multi-resolution image"""

    @abstractmethod
    def level_axes(self, level: int) -> Axes:
        """Return the axes for the given level."""

    @abstractmethod
    def level_image(self, level: int) -> np.ndarray:
        """
        Return the image for the given level as numpy array.

        The axes of the array are specified by `level_axes(level)`
        """

    @abstractmethod
    def level_metadata(self, level: int) -> Dict[str, Any]:
        """Return the metadata for the given level."""

    @property
    @abstractmethod
    def group_metadata(self) -> Dict[str, Any]:
        """Return the metadata for the whole multi-resolution image."""


class ImageWriter(ABC):
    def __enter__(self) -> ImageWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @abstractmethod
    def write_level_array(self, level: int, array: tiledb.Array) -> None:
        """Write the image at the given level."""

    @abstractmethod
    def write_group_metadata(self, group: tiledb.Group) -> None:
        """Write the metadata for the whole multi-resolution image."""


class ImageConverter(ABC):
    def __init__(
        self,
        c_dim: Dimension = Dimension("C", 3),  # channel
        y_dim: Dimension = Dimension("Y", 1024),  # height
        x_dim: Dimension = Dimension("X", 1024),  # width
    ):
        self._dims = (c_dim, y_dim, x_dim)

    def from_tiledb(
        self, input_path: str, output_path: str, level_min: int = 0
    ) -> None:
        """
        Convert a TileDB Group of Arrays back to other format images, one per level.

        :param input_path: path to the TileDB group of arrays
        :param output_path: path to the image
        :param level_min: minimum level of the image to be converted. By default set to 0
            to convert all levels.
        """
        # open all level arrays, keep those with level >= level_min and sort them by level
        level_arrays = []
        group = tiledb.Group(input_path, "r")
        for member in group:
            array = tiledb.open(member.uri)
            level = array.meta.get("level", 0)
            if level < level_min:
                array.close()
                continue
            level_arrays.append((level, array))
        level_arrays.sort(key=itemgetter(0))

        with self._get_image_writer(output_path) as writer:
            writer.write_group_metadata(group)
            for level, array in level_arrays:
                writer.write_level_array(level, array)
                array.close()

    def to_tiledb(
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
        with self._get_image_reader(input_path) as reader:
            # Create a TileDB array for each level in range(level_min, reader.level_count)
            uris = []
            for level in range(level_min, reader.level_count):
                uri = os.path.join(output_group_path, f"l_{level}.tdb")
                image = reader.level_image(level)
                canonical_image = reader.level_axes(level).transpose(image)
                level_metadata = reader.level_metadata(level)
                level_metadata["level"] = level
                self._write_image(uri, canonical_image, level_metadata)
                uris.append(uri)
            # Write group metadata
            with tiledb.Group(output_group_path, "w") as group:
                group.meta.update(reader.group_metadata)
                for level_uri in uris:
                    if urlparse(level_uri).scheme == "tiledb":
                        group.add(level_uri, relative=False)
                    else:
                        group.add(os.path.basename(level_uri), relative=True)

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

    @abstractmethod
    def _get_image_writer(self, output_path: str) -> ImageWriter:
        """Return an ImageWriter for the given input path."""

    @abstractmethod
    def _get_image_reader(self, input_path: str) -> ImageReader:
        """Return an ImageReader for the given input path."""
