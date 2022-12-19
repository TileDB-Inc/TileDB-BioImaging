from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import ChainMap
from operator import itemgetter
from typing import Any, Dict, Mapping, Optional, Tuple, Type
from urllib.parse import urlparse

import numpy as np

import tiledb

from .axes import Axes, AxesMapper
from .tiles import iter_tiles


class ImageReader(ABC):
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
    def level_dtype(self, level: int) -> np.dtype:
        """Return the dtype of the image for the given level."""

    @abstractmethod
    def level_shape(self, level: int) -> Tuple[int, ...]:
        """Return the shape of the image for the given level."""

    @abstractmethod
    def level_image(
        self, level: int, tile: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """
        Return the image for the given level as numpy array.

        The axes of the array are specified by the `axes` property.

        :param tile: If not None, a tuple of slices (one per each axes) that specify the
            subregion of the image to return.
        """

    @abstractmethod
    def level_metadata(self, level: int) -> Dict[str, Any]:
        """Return the metadata for the given level."""

    @property
    @abstractmethod
    def group_metadata(self) -> Dict[str, Any]:
        """Return the metadata for the whole multi-resolution image."""


class ImageWriter(ABC):
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


class ImageConverter:
    _DEFAULT_TILES = {"T": 1, "C": 3, "Z": 1, "Y": 1024, "X": 1024}
    _ImageReaderType: Optional[Type[ImageReader]] = None
    _ImageWriterType: Optional[Type[ImageWriter]] = None

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
        if cls._ImageWriterType is None:
            raise NotImplementedError(f"{cls} does not support exporting")

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

        with cls._ImageWriterType(output_path) as writer:
            writer.write_group_metadata(group.meta)
            original_axes = Axes(group.meta["axes"])
            for level, array in level_arrays:
                # read image and transform to the original axes
                stored_axes = Axes(dim.name for dim in array.domain)
                image = AxesMapper(stored_axes, original_axes).map_array(array[:])
                # write image and close the array
                writer.write_level_image(level, image, array.meta)
                array.close()

    @classmethod
    def to_tiledb(
        cls,
        input_path: str,
        output_path: str,
        *,
        level_min: int = 0,
        tiles: Optional[Mapping[str, int]] = None,
        preserve_axes: bool = False,
        chunked: bool = False,
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
        :param chunked: If true, convert one image tile at a time instead of the whole image.
        """
        if cls._ImageReaderType is None:
            raise NotImplementedError(f"{cls} does not support importing")

        if tiledb.object_type(output_path) != "group":
            tiledb.group_create(output_path)

        with cls._ImageReaderType(input_path) as reader:
            input_axes = reader.axes
            # Create a TileDB array for each level in range(level_min, reader.level_count)
            uris = []
            for level in range(level_min, reader.level_count):
                uri = os.path.join(output_path, f"l_{level}.tdb")
                if tiledb.object_type(uri) == "array":
                    # level has already been converted
                    continue

                # read metadata and image
                metadata = reader.level_metadata(level)
                level_dtype = reader.level_dtype(level)
                level_shape = reader.level_shape(level)

                # determine level axes and (potentially) transformed level shape
                if preserve_axes:
                    level_axes = input_axes
                else:
                    level_axes = input_axes.canonical(level_shape)
                axes_mapper = AxesMapper(input_axes, level_axes)
                level_shape = axes_mapper.map_shape(level_shape)

                # create TileDB array
                schema = _get_schema(
                    axes=level_axes,
                    shape=level_shape,
                    attr_dtype=level_dtype,
                    max_tiles=ChainMap(dict(tiles or {}), cls._DEFAULT_TILES),
                )
                tiledb.Array.create(uri, schema)

                # write image and metadata to TileDB array
                with tiledb.open(uri, "w") as a:
                    a.meta.update(metadata, level=level)
                    if chunked:
                        inv_axes_mapper = AxesMapper(level_axes, input_axes)
                        for level_tile in iter_tiles(a.domain):
                            input_tile = inv_axes_mapper.map_tile(level_tile)
                            image = reader.level_image(level, input_tile)
                            a[level_tile] = axes_mapper.map_array(image)
                    else:
                        image = reader.level_image(level)
                        a[:] = axes_mapper.map_array(image)
                uris.append(uri)

            # Write group metadata
            with tiledb.Group(output_path, "w") as group:
                group.meta.update(reader.group_metadata, axes=input_axes.dims)
                for level_uri in uris:
                    if urlparse(level_uri).scheme == "tiledb":
                        group.add(level_uri, relative=False)
                    else:
                        group.add(os.path.basename(level_uri), relative=True)


def _get_schema(
    axes: Axes,
    shape: Tuple[int, ...],
    attr_dtype: np.dtype,
    max_tiles: Mapping[str, int],
) -> tiledb.ArraySchema:
    # find the smallest dtype that can hold `np.prod(shape)` values
    dim_dtype = np.min_scalar_type(np.prod(shape))
    dims = []
    assert len(axes.dims) == len(shape)
    for dim_name, dim_size in zip(axes.dims, shape):
        dims.append(
            tiledb.Dim(
                dim_name,
                domain=(0, dim_size - 1),
                dtype=dim_dtype,
                tile=min(dim_size, max_tiles[dim_name]),
            )
        )
    return tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        attrs=[
            tiledb.Attr(
                name="",
                dtype=attr_dtype,
                filters=[tiledb.ZstdFilter(level=0)],
            )
        ],
    )
