from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import ChainMap
from operator import itemgetter
from typing import Any, Dict, Mapping, Optional, Tuple, Type
from urllib.parse import urlparse

import numpy as np

import tiledb

from ..compressor_factory import (
    CompressorArguments,
    T,
    WebpArguments,
    ZstdArguments,
    createCompressor,
)
from .axes import Axes, transpose_array


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
                # read image and transpose to the original axes
                # stored_axes = Axes(dim.name for dim in array.domain)
                # print(f"Stored axes {stored_axes}")
                # image = transpose_array(array[:], stored_axes.dims, original_axes.dims)
                # print(f"Image shape {image.shape}")

                # stored_axes = Axes(dim.name for dim in array.domain)
                # print(f"Stored axes {stored_axes}")
                if isinstance(array.attr(0).filters[0], tiledb.filter.WebpFilter):
                    channels = (
                        3
                        if int(array.attr(0).filters[0].input_format)
                        < int(tiledb.filter.lt.WebpInputFormat.WEBP_RGBA)
                        else 4
                    )
                    image = array[:]
                    image = np.reshape(
                        image, (-1, image.shape[1] // channels, channels)
                    )
                    image = transpose_array(image, "YXC", original_axes.dims)
                else:
                    # read image and transpose to the original axes
                    stored_axes = Axes(dim.name for dim in array.domain)
                    image = transpose_array(
                        array[:], stored_axes.dims, original_axes.dims
                    )

                # write image and close   the array
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
        compressor_arguments: CompressorArguments[T] = ZstdArguments(level=0),
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
        :param compressor_arguments: If None no compression filter is used
        """
        if cls._ImageReaderType is None:
            raise NotImplementedError(f"{cls} does not support importing")

        if tiledb.object_type(output_path) != "group":
            tiledb.group_create(output_path)
        with cls._ImageReaderType(input_path) as reader:
            axes = reader.axes
            # Create a TileDB array for each level in range(level_min, reader.level_count)
            uris = []
            for level in range(level_min, reader.level_count):
                uri = os.path.join(output_path, f"l_{level}.tdb")
                if tiledb.object_type(uri) == "array":
                    # level has already been converted
                    continue

                # read metadata and image
                metadata = reader.level_metadata(level)
                image = reader.level_image(level)
                level_dtype = reader.level_dtype(level)
                level_shape = reader.level_shape(level)

                # determine axes and (optionally) transpose image to canonical axes
                level_axes = axes if preserve_axes else axes.canonical(level_shape)
                if level_axes != axes:
                    image = transpose_array(image, axes.dims, level_axes.dims)
                    level_shape = image.shape

                if isinstance(compressor_arguments, WebpArguments):

                    if level_dtype != np.uint8:
                        raise ValueError(
                            f"WebP compressor in {cls} does not support {level_dtype} data type."
                        )

                    if any(dim in level_axes.dims for dim in "TZ"):
                        raise NotImplementedError(
                            f"WebP compressor in {cls} does not support T ot Z dimensions"
                        )

                    image = transpose_array(image, level_axes.dims, "YXC")
                    level_axes = Axes("YXC")
                    level_shape = image.shape

                    if image.shape[2] != 3 and image.shape[2] != 4:
                        raise NotImplementedError(
                            f"WebP compressor in {cls} does not support images with {image.shape[2]} channels"
                        )

                    compressor_arguments.image_format = (
                        tiledb.filter.lt.WebpInputFormat.WEBP_RGB
                        if image.shape[2] == 3
                        else tiledb.filter.lt.WebpInputFormat.WEBP_RGBA
                    )

                # create TileDB array
                schema = _get_schema(
                    axes=level_axes,
                    shape=level_shape,
                    attr_dtype=level_dtype,
                    max_tiles=ChainMap(dict(tiles or {}), cls._DEFAULT_TILES),
                    compression_arguments=compressor_arguments,
                )
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


def _get_schema(
    axes: Axes,
    shape: Tuple[int, ...],
    attr_dtype: np.dtype,
    max_tiles: Mapping[str, int],
    compression_arguments: CompressorArguments[T],
) -> tiledb.ArraySchema:
    # find the smallest dtype that can hold the number of image scalar values
    dim_dtype = np.min_scalar_type(np.prod(shape))
    dims = []
    assert len(axes.dims) == len(shape)
    if isinstance(compression_arguments, WebpArguments):
        assert axes.dims == "YXC"
        dims.append(
            tiledb.Dim(
                "Y",
                domain=(0, shape[0] - 1),
                dtype=dim_dtype,
                tile=min(shape[0], max_tiles["Y"]),
            )
        )
        dims.append(
            tiledb.Dim(
                "X",
                domain=(0, shape[1] * shape[2] - 1),
                dtype=dim_dtype,
                tile=min(shape[1], max_tiles["X"]) * shape[2],
            )
        )
    else:
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
                filters=[createCompressor(compression_arguments)],
            )
        ],
    )
