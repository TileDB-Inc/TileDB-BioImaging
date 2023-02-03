from __future__ import annotations

import json
import os
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type
from urllib.parse import urlparse

import numpy as np
from tqdm import tqdm

from .scale import Scaler

try:
    from tiledb.cloud.groups import register as register_group
except ImportError:
    register_group = None

import tiledb

from ..openslide import TileDBOpenSlide, get_pixel_depth
from ..version import version
from . import DATASET_TYPE, FMT_VERSION
from .axes import Axes, AxesMapper
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
        input_path: str,
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

        :param input_path: path to the input image
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
        if cls._ImageReaderType is None:
            raise NotImplementedError(f"{cls} does not support importing")

        pixel_depth = get_pixel_depth(compressor)
        max_tiles = cls._DEFAULT_TILES.copy()
        max_tiles.update(tiles)
        max_tiles["X"] *= pixel_depth

        rw_group = _ReadWriteGroup(output_path)
        reader = cls._ImageReaderType(input_path, **reader_kwargs)

        with rw_group, reader:
            if (
                rw_group.r_group.meta["pkg_version"] != version
                or rw_group.r_group.meta["fmt_version"] != FMT_VERSION
            ):
                raise EnvironmentError(
                    "Incremental ingestion is supported only between same format and package versions"
                )

            if pyramid_kwargs is not None:
                level_max = level_min + 1
                if reader.level_count > level_max:
                    warnings.warn(
                        f"The image contains multiple levels but only level {level_min} "
                        "will be considered for generating the image pyramid"
                    )
            else:
                level_max = reader.level_count

            source_axes = reader.axes
            for level in range(level_min, level_max):
                # create mapper from source to target axes
                source_shape = reader.level_shape(level)
                if pixel_depth == 1:
                    if preserve_axes:
                        target_axes = source_axes
                    else:
                        target_axes = source_axes.canonical(source_shape)
                    axes_mapper = AxesMapper(source_axes, target_axes)
                    dim_names = tuple(target_axes.dims)
                else:
                    raise NotImplementedError

                # create TileDB schema
                dim_shape = axes_mapper.map_shape(source_shape)
                attr_dtype = reader.level_dtype(level)
                schema = _get_schema(
                    dim_names, dim_shape, max_tiles, attr_dtype, compressor
                )

                # get or create TileDB array uri
                name = f"l_{level}.tdb"
                uri, created = rw_group.get_or_create(name, schema)
                if not created:
                    continue

                # write image and metadata to TileDB array
                with tiledb.open(uri, "w") as out_array:
                    _convert_level_to_tiledb(
                        reader, level, out_array, axes_mapper, chunked, max_workers
                    )

            if pyramid_kwargs is not None:
                _create_image_pyramid(
                    rw_group, uri, level_min, max_tiles, compressor, pyramid_kwargs
                )

        with rw_group:
            levels_meta = _compute_levels_meta(rw_group)

            # Write group metadata
            rw_group.w_group.meta.update(
                reader.group_metadata,
                axes=source_axes.dims,
                pkg_version=version,
                fmt_version=FMT_VERSION,
                dataset_type=DATASET_TYPE,
                levels=json.dumps(levels_meta),
            )

        if register_group is not None and urlparse(output_path).scheme == "tiledb":
            register_group(name=os.path.basename(output_path), **register_kwargs)


class _ReadWriteGroup:
    def __init__(self, uri: str):
        if tiledb.object_type(uri) != "group":
            tiledb.group_create(uri)
        self.uri = uri

    def __enter__(self) -> _ReadWriteGroup:
        self.r_group = tiledb.Group(self.uri, "r")
        self.w_group = tiledb.Group(self.uri, "w")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.r_group.close()
        self.w_group.close()

    def get_or_create(self, name: str, schema: tiledb.ArraySchema) -> Tuple[str, bool]:
        create = False
        if name in self.r_group:
            uri = self.r_group[name].uri
        else:
            uri = os.path.join(self.r_group.uri, name)
            if not tiledb.array_exists(uri):
                tiledb.Array.create(uri, schema)
                create = True
            else:
                # The array exists but it's not added as group member with the given name.
                # It is possible though that it was added as an anonymous member.
                # In this case we should remove the member, using as key either the uri
                # (if added with relative=False) or the name (if added with relative=True).
                for ref in uri, name:
                    try:
                        self.w_group.remove(ref)
                    except tiledb.TileDBError:
                        pass
                    else:
                        # Attempting to remove and then re-add a member with the same name
                        # fails with "[TileDB::Group] Error: Cannot add group member,
                        # member already set for removal.". To work around this we need to
                        # close the write group (to flush the removal) and and reopen it
                        # (to allow the add operation)
                        self.w_group.close()
                        self.w_group.open("w")
            # register the uri with the given name
            self.w_group.add(uri, name)
        return uri, create


def _get_schema(
    dim_names: Tuple[str, ...],
    dim_shape: Tuple[int, ...],
    max_tiles: Mapping[str, int],
    attr_dtype: np.dtype,
    compressor: tiledb.Filter,
) -> tiledb.ArraySchema:
    # find the smallest dtype that can hold `np.prod(dim_shape)` values
    dim_dtype = np.min_scalar_type(np.prod(dim_shape))

    dims = []
    assert len(dim_names) == len(dim_shape), (dim_names, dim_shape)
    for dim_name, dim_size in zip(dim_names, dim_shape):
        dim_tile = min(dim_size, max_tiles[dim_name])
        dim = tiledb.Dim(dim_name, (0, dim_size - 1), dim_tile, dtype=dim_dtype)
        dims.append(dim)
    attr = tiledb.Attr(name="", dtype=attr_dtype, filters=[compressor])
    return tiledb.ArraySchema(domain=tiledb.Domain(*dims), attrs=[attr])


def _convert_level_to_tiledb(
    reader: ImageReader,
    level: int,
    out_array: tiledb.Array,
    axes_mapper: AxesMapper,
    chunked: bool,
    max_workers: int,
) -> None:
    out_array.meta.update(reader.level_metadata(level), level=level)
    if chunked or max_workers:
        inv_axes_mapper = axes_mapper.inverted

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


def _create_image_pyramid(
    rw_group: _ReadWriteGroup,
    base_uri: str,
    base_level: int,
    max_tiles: Mapping[str, int],
    compressor: tiledb.Filter,
    pyramid_kwargs: Mapping[str, Any],
) -> None:
    with tiledb.open(base_uri, "r") as a:
        base_shape = a.shape
        dim_names = tuple(dim.name for dim in a.domain)
        dim_axes = "".join(dim_names)
        attr_dtype = a.attr(0).dtype

    scaler = Scaler(base_shape, dim_axes, **pyramid_kwargs)
    for i, dim_shape in enumerate(scaler.level_shapes):
        level = base_level + 1 + i
        name = f"l_{level}.tdb"
        schema = _get_schema(dim_names, dim_shape, max_tiles, attr_dtype, compressor)
        uri, created = rw_group.get_or_create(name, schema)
        if not created:
            continue

        with tiledb.open(uri, "w") as out_array:
            out_array.meta.update(level=level)
            with tiledb.open(base_uri, "r") as in_array:
                scaler.apply(in_array, out_array, i)

        # if a non-progressive method is used, the input layer of the scaler
        # is the base image layer else we use the previously generated layer
        if scaler.progressive:
            base_uri = uri


def _compute_levels_meta(rw_group: _ReadWriteGroup) -> Sequence[Mapping[str, Any]]:
    levels_meta: List[Mapping[str, Any]] = []
    for group_obj in rw_group.r_group:
        if group_obj.type == tiledb.Array:
            with tiledb.open(group_obj.uri) as lvl_array:
                domain = lvl_array.schema.domain
                dim_names = tuple(
                    [domain.dim(dim_idx).name for dim_idx in range(domain.ndim)]
                )
                dim_axes = "".join(dim_names)
                levels_meta.append(
                    dict(
                        level=lvl_array.meta["level"],
                        name=os.path.basename(lvl_array.uri),
                        axes=dim_axes,
                        shape=lvl_array.shape,
                    )
                )
        else:
            # Another group or invalid
            continue
    return levels_meta
