from __future__ import annotations

import json
import os
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, Union
from urllib.parse import urlparse

import numpy as np
from tqdm import tqdm

try:
    from tiledb.cloud.groups import register as register_group
except ImportError:
    register_group = None

import tiledb
from tiledb.cc import WebpInputFormat

from ..helpers import (
    ReadWriteGroup,
    create_image_pyramid,
    get_pixel_depth,
    get_schema,
    iter_levels_meta,
    open_bioimg,
)
from ..openslide import TileDBOpenSlide
from ..version import version as PKG_VERSION
from . import DATASET_TYPE, FMT_VERSION
from .axes import Axes
from .tiles import iter_tiles, num_tiles


class ImageReader(ABC):
    @abstractmethod
    def __init__(self, input_path: str, **kwargs: Any):
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
    def channels(self) -> Sequence[str]:
        """Names of the channels (C axis) of this multi-resolution image."""

    @property
    def webp_format(self) -> WebpInputFormat:
        """WebpInputFormat of this multi-resolution image. Defaults to WEBP_NONE."""
        return WebpInputFormat.WEBP_NONE

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
    # setting a tile to "infinite" effectively makes it equal to the dimension size
    _DEFAULT_TILES = {"T": 1, "C": np.inf, "Z": 1, "Y": 1024, "X": 1024}
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

        slide = TileDBOpenSlide(input_path)
        writer = cls._ImageWriterType(output_path)
        with slide, writer:
            writer.write_group_metadata(slide.properties)
            for level in slide.levels:
                if level < level_min:
                    continue
                level_image = slide.read_level(level, to_original_axes=True)
                level_metadata = slide.level_properties(level)
                writer.write_level_image(level, level_image, level_metadata)

    @classmethod
    def to_tiledb(
        cls,
        source: Union[str, ImageReader],
        output_path: str,
        *,
        level_min: int = 0,
        tiles: Mapping[str, int] = {},
        preserve_axes: bool = False,
        chunked: bool = False,
        max_workers: int = 0,
        compressor: tiledb.Filter = tiledb.ZstdFilter(level=0),
        register_kwargs: Mapping[str, Any] = {},
        reader_kwargs: Mapping[str, Any] = {},
        pyramid_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Convert an image to a TileDB Group of Arrays, one per level.

        :param source: path to the input image or ImageReader object
        :param output_path: path to the TileDB group of arrays
        :param level_min: minimum level of the image to be converted. By default set to 0
            to convert all levels.
        :param tiles: A mapping from dimension name (one of 'T', 'C', 'Z', 'Y', 'X') to
            the (maximum) tile for this dimension.
        :param preserve_axes: If true, preserve the axes order of the original image.
        :param chunked: If true, convert one tile at a time instead of the whole image.
            **Note**: The OpenSlideConverter may not be 100% lossless with chunked=True
            for levels>0, even though the converted images look visually identical to the
            original ones.
        :param max_workers: Maximum number of threads that can be used for conversion.
            Applicable only if chunked=True.
        :param compressor: TileDB compression filter
        :param register_kwargs: Cloud group registration optional args e.g namespace,
            parent_uri, storage_uri, credentials_name
        :param reader_kwargs: Keyword arguments passed to the _ImageReaderType constructor.
        :param pyramid_kwargs: Keyword arguments passed to the scaler constructor for
            generating downsampled versions of the base level. Valid keyword arguments are:
            scale_factors (Required): The downsampling factor for each level
            scale_axes (Optional): Default "XY". The axes which will be downsampled
            chunked (Optional): Default False. If true the image is split into chunks and
                each one is independently downsampled. If false the entire image is
                downsampled at once, but it requires more memory.
            progressive (Optional): Default False. If true each downsampled image is
                generated using the previous level. If false for every downsampled image
                the level_min is used, but it requires more memory.
            order (Optional): Default 1. The order of the spline interpolation. The order
                has to be in the range 0-5. See `skimage.transform.warp` for detail.
            max_workers (Optional): Default None. The maximum number of workers for
                chunked downsampling. If None, it will default to the number of processors
                on the machine, multiplied by 5.
        """
        if isinstance(source, ImageReader):
            if cls._ImageReaderType != source.__class__:
                raise ValueError("Image reader should match converter on source format")
            reader = source
        elif cls._ImageReaderType is not None:
            reader = cls._ImageReaderType(source, **reader_kwargs)
        else:
            raise NotImplementedError(f"{cls} does not support importing")

        max_tiles = cls._DEFAULT_TILES.copy()
        max_tiles.update(tiles)

        rw_group = ReadWriteGroup(output_path)
        with rw_group, reader:
            stored_pkg_version = rw_group.r_group.meta.get("pkg_version")
            if stored_pkg_version not in (None, PKG_VERSION):
                raise RuntimeError(
                    "Incremental ingestion is not supported for different versions: "
                    f"current version is {PKG_VERSION}, stored version is {stored_pkg_version}"
                )

            if (
                isinstance(compressor, tiledb.WebpFilter)
                and compressor.input_format == WebpInputFormat.WEBP_NONE
            ):
                compressor = tiledb.WebpFilter(
                    input_format=reader.webp_format,
                    quality=compressor.quality,
                    lossless=compressor.lossless,
                )

            convert_kwargs = dict(
                reader=reader,
                rw_group=rw_group,
                max_tiles=max_tiles,
                preserve_axes=preserve_axes,
                chunked=chunked,
                max_workers=max_workers,
                compressor=compressor,
            )
            if pyramid_kwargs is not None:
                if level_min < reader.level_count - 1:
                    warnings.warn(
                        f"The image contains multiple levels but only level {level_min} "
                        "will be considered for generating the image pyramid"
                    )
                uri = _convert_level_to_tiledb(level_min, **convert_kwargs)
                create_image_pyramid(
                    rw_group, uri, level_min, max_tiles, compressor, pyramid_kwargs
                )
            else:
                for level in range(level_min, reader.level_count):
                    _convert_level_to_tiledb(level, **convert_kwargs)

        with rw_group:
            rw_group.w_group.meta.update(
                reader.group_metadata,
                axes=reader.axes.dims,
                pixel_depth=get_pixel_depth(compressor),
                pkg_version=PKG_VERSION,
                fmt_version=FMT_VERSION,
                dataset_type=DATASET_TYPE,
                channels=json.dumps(reader.channels),
                levels=json.dumps(
                    sorted(iter_levels_meta(rw_group.r_group), key=itemgetter("level"))
                ),
            )

        if register_group is not None and urlparse(output_path).scheme == "tiledb":
            register_group(name=os.path.basename(output_path), **register_kwargs)


def _convert_level_to_tiledb(
    level: int,
    *,
    reader: ImageReader,
    rw_group: ReadWriteGroup,
    max_tiles: Dict[str, int],
    preserve_axes: bool,
    chunked: bool,
    max_workers: int,
    compressor: tiledb.Filter,
) -> str:
    # create mapper from source to target axes
    source_axes = reader.axes
    source_shape = reader.level_shape(level)
    pixel_depth = get_pixel_depth(compressor)
    if pixel_depth == 1:
        if preserve_axes:
            target_axes = source_axes
        else:
            target_axes = source_axes.canonical(source_shape)
        axes_mapper = source_axes.mapper(target_axes)
        dim_names = tuple(target_axes.dims)
    else:
        max_tiles["X"] *= pixel_depth
        axes_mapper = source_axes.webp_mapper(pixel_depth)
        dim_names = ("Y", "X")

    # create TileDB schema
    dim_shape = axes_mapper.map_shape(source_shape)
    attr_dtype = reader.level_dtype(level)
    schema = get_schema(dim_names, dim_shape, max_tiles, attr_dtype, compressor)

    # get or create TileDB array uri
    uri, created = rw_group.get_or_create(f"l_{level}.tdb", schema)
    if created:
        # write image and metadata to TileDB array
        with open_bioimg(uri, "w") as out_array:
            out_array.meta.update(reader.level_metadata(level), level=level)
            if chunked or max_workers:
                inv_axes_mapper = axes_mapper.inverse

                def tile_to_tiledb(level_tile: Tuple[slice, ...]) -> None:
                    source_tile = inv_axes_mapper.map_tile(level_tile)
                    image = reader.level_image(level, source_tile)
                    out_array[level_tile] = axes_mapper.map_array(image)

                ex = ThreadPoolExecutor(max_workers) if max_workers else None
                mapper = getattr(ex, "map", map)
                for _ in tqdm(
                    mapper(tile_to_tiledb, iter_tiles(out_array.domain)),
                    desc=f"Ingesting level {level}",
                    total=num_tiles(out_array.domain),
                    unit="tiles",
                ):
                    pass
                if ex:
                    ex.shutdown()
            else:
                image = reader.level_image(level)
                out_array[:] = axes_mapper.map_array(image)
    return uri
