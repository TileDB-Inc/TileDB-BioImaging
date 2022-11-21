from __future__ import annotations

import os
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Any, Dict, Mapping
from urllib.parse import urlparse

import numpy as np

import tiledb

from .axes import Axes, transpose_array


class ImageReader(ABC):
    _DEFAULT_TILES = {"T": 1, "C": 3, "Z": 1, "Y": 1024, "X": 1024}

    @classmethod
    def to_tiledb(
        cls,
        input_path: str,
        output_path: str,
        *,
        level_min: int = 0,
        tiles: Mapping[str, int] = {},
        preserve_axes: bool = False,
    ) -> None:
        """
        Convert an image to a TileDB Group of Arrays, one per level.

        :param input_path: path to the input image
        :param output_path: path to the TileDB group of arrays
        :param level_min: minimum level of the image to be converted. By default set to 0
            to convert all levels.
        :param tiles: A mapping from dimension name (one of 'T', 'C', 'Z', 'Y', 'X') to
            the (maximum) tile for this dimension.
        :param preserve_axes: If true, preserve the axes order of the original image.
        """
        tiledb.group_create(output_path)
        with cls(input_path) as reader:
            axes = reader.axes
            # Create a TileDB array for each level in range(level_min, reader.level_count)
            uris = []
            for level in range(level_min, reader.level_count):
                # read metadata and image
                metadata = reader.level_metadata(level)
                image = reader.level_image(level)
                # determine axes and (optionally) transpose image to canonical axes
                if preserve_axes:
                    level_axes = axes
                else:
                    level_axes = axes.canonical(image)
                    image = transpose_array(image, axes.dims, level_axes.dims)
                # create TileDB array
                uri = os.path.join(output_path, f"l_{level}.tdb")
                schema = cls._get_schema(image, level_axes, tiles)
                tiledb.Array.create(uri, schema)
                # write image and metadata to TileDB array
                with tiledb.open(uri, "w") as a:
                    a[:] = image
                    a.meta.update(metadata, level=level)
                uris.append(uri)

            # Write group metadata
            with tiledb.Group(output_path, "w") as group:
                group.meta.update(reader.group_metadata, axes=axes.dims)
                for level_uri in uris:
                    if urlparse(level_uri).scheme == "tiledb":
                        group.add(level_uri, relative=False)
                    else:
                        group.add(os.path.basename(level_uri), relative=True)

    @classmethod
    def _get_schema(
        cls, image: np.ndarray, axes: Axes, tiles: Mapping[str, int]
    ) -> tiledb.ArraySchema:
        # find the smallest dtype that can hold the number of image scalar values
        dim_dtype = np.min_scalar_type(image.size)
        dims = []
        assert len(axes.dims) == len(image.shape)
        for dim_name, dim_size in zip(axes.dims, image.shape):
            max_tile = tiles.get(dim_name, cls._DEFAULT_TILES[dim_name])
            dims.append(
                tiledb.Dim(
                    dim_name,
                    domain=(0, dim_size - 1),
                    dtype=dim_dtype,
                    tile=min(dim_size, max_tile),
                )
            )
        return tiledb.ArraySchema(
            domain=tiledb.Domain(*dims),
            attrs=[
                tiledb.Attr(
                    name="",
                    dtype=image.dtype,
                    filters=[tiledb.ZstdFilter(level=0)],
                )
            ],
        )

    @abstractmethod
    def __init__(self, input_path: str):
        """Initialize this ImageReader"""

    def __enter__(self) -> ImageReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @property
    @abstractmethod
    def axes(self) -> Axes:
        """The axes of this multi-resolution image."""

    @property
    @abstractmethod
    def level_count(self) -> int:
        """
        The number of levels for this multi-resolution image.

        Levels are numbered from 0 (highest resolution) to level_count - 1 (lowest resolution).
        """

    @abstractmethod
    def level_image(self, level: int) -> np.ndarray:
        """
        Return the image for the given level as numpy array.

        The axes of the array are specified by the `axes` property.
        """

    @abstractmethod
    def level_metadata(self, level: int) -> Dict[str, Any]:
        """Return the metadata for the given level."""

    @property
    @abstractmethod
    def group_metadata(self) -> Dict[str, Any]:
        """Return the metadata for the whole multi-resolution image."""


class ImageWriter(ABC):
    @classmethod
    def from_tiledb(
        cls, input_path: str, output_path: str, *, level_min: int = 0
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

        with cls(output_path) as writer:
            writer.write_group_metadata(group.meta)
            original_axes = Axes(group.meta["axes"])
            for level, array in level_arrays:
                # read image and transpose to the original axes
                stored_axes = Axes(dim.name for dim in array.domain)
                image = transpose_array(array[:], stored_axes.dims, original_axes.dims)
                # write image and close the array
                writer.write_level_image(level, image, array.meta)
                array.close()

    @abstractmethod
    def __init__(self, output_path: str):
        """Initialize this ImageWriter"""

    def __enter__(self) -> ImageWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @abstractmethod
    def write_group_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Write metadata for the whole multi-resolution image."""

    @abstractmethod
    def write_level_image(
        self, level: int, image: np.ndarray, metadata: Mapping[str, Any]
    ) -> None:
        """
        Write the image for the given level.

        :param level: Number corresponding to a level
        :param image: Image for the given level as numpy array
        :param metadata: Metadata for the given level
        """
