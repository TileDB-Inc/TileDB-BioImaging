from __future__ import annotations

import os
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Any, Dict, Mapping
from urllib.parse import urlparse

import numpy as np

import tiledb

from .axes import Axes


class ImageReader(ABC):
    def __enter__(self) -> ImageReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Return the number of levels for this multi-resolution image.
        Levels are numbered from 0 (highest resolution) to level_count - 1 (lowest resolution).

        :return: The number of levels in the slide
        """

    @abstractmethod
    def level_axes(self, level: int) -> Axes:
        """Return the axes for the given level.

        :param level: Number corresponding to a level

        :return: Axes object containing the axes of the given level.
        """

    @abstractmethod
    def level_image(self, level: int) -> np.ndarray:
        """
        Return the image for the given level as numpy array.
        The axes of the array are specified by `level_axes(level)`

        :param level: Number corresponding to a level

        :return: np.ndarray of the image on the level given
        """

    @abstractmethod
    def level_metadata(self, level: int) -> Dict[str, Any]:
        """Return the metadata for the given level.

        :param level: Number corresponding to a level

        :return: A Dict containing the metadata of the given level
        """

    @property
    @abstractmethod
    def group_metadata(self) -> Dict[str, Any]:
        """Return the metadata for the whole multi-resolution image.

        :return: A Dict containing the metadata of the image
        """


class ImageWriter(ABC):
    def __enter__(self) -> ImageWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @abstractmethod
    def write_level_array(self, level: int, array: tiledb.Array) -> None:
        """
        Writes the resolution image of the level given from a TileDB array

        :param level: Number corresponding to a level
        :param array: tiledb.Array containing the data of the level
        """

    @abstractmethod
    def write_group_metadata(self, group: tiledb.Group) -> None:
        """
        Writes metadata of the image

        :param group: tiledb.Group that contains the image
        """


class ImageConverter(ABC):

    _DEFAULT_TILES = {"T": 1, "C": 3, "Z": 1, "Y": 1024, "X": 1024}

    def from_tiledb(
        self, input_path: str, output_path: str, *, level_min: int = 0
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
        self,
        input_path: str,
        output_group_path: str,
        *,
        level_min: int = 0,
        tiles: Mapping[str, int] = {},
        preserve_axes: bool = False,
    ) -> None:
        """
        Convert an image to a TileDB Group of Arrays, one per level.

        :param input_path: path to the input image
        :param output_group_path: path to the TileDB group of arrays
        :param level_min: minimum level of the image to be converted. By default set to 0
            to convert all levels.
        :param tiles: A mapping from dimension name (one of 'T', 'C', 'Z', 'Y', 'X') to
            the (maximum) tile for this dimension.
        :param preserve_axes: If true, preserve the axes order of the original image.
        """
        tiledb.group_create(output_group_path)
        with self._get_image_reader(input_path) as reader:
            # Create a TileDB array for each level in range(level_min, reader.level_count)
            uris = []
            for level in range(level_min, reader.level_count):
                # read and update metadata
                metadata = reader.level_metadata(level)
                metadata["level"] = level
                # read axes and image
                axes = reader.level_axes(level)
                image = reader.level_image(level)
                if not preserve_axes:
                    # transpose image to canonical axes
                    canonical_axes = axes.canonical(image)
                    image = axes.transpose(image, canonical_axes)
                    axes = canonical_axes
                # create TileDB array
                uri = os.path.join(output_group_path, f"l_{level}.tdb")
                schema = self._get_schema(image, axes, tiles)
                tiledb.Array.create(uri, schema)
                # write image and metadata to TileDB array
                with tiledb.open(uri, "w") as a:
                    a[:] = image
                    a.meta.update(metadata)
                uris.append(uri)

            # Write group metadata
            with tiledb.Group(output_group_path, "w") as group:
                group.meta.update(reader.group_metadata)
                for level_uri in uris:
                    if urlparse(level_uri).scheme == "tiledb":
                        group.add(level_uri, relative=False)
                    else:
                        group.add(os.path.basename(level_uri), relative=True)

    def _get_schema(
        self, image: np.ndarray, axes: Axes, tiles: Mapping[str, int]
    ) -> tiledb.ArraySchema:
        # find the smallest dtype that can hold the number of image scalar values
        dim_dtype = np.min_scalar_type(image.size)
        dims = []
        assert len(axes.dims) == len(image.shape)
        for dim_name, dim_size in zip(axes.dims, image.shape):
            max_tile = tiles.get(dim_name, self._DEFAULT_TILES[dim_name])
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
    def _get_image_writer(self, output_path: str) -> ImageWriter:
        """Return an ImageWriter for the given input path."""

    @abstractmethod
    def _get_image_reader(self, input_path: str) -> ImageReader:
        """Return an ImageReader for the given input path."""
