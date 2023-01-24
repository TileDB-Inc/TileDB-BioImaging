from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Mapping, Optional, Tuple, Type
from operator import itemgetter
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type
from urllib.parse import urlparse

import numpy as np
from tqdm import tqdm

from ..version import version

from .scale import ScaleMethod, Scaler, ScalerMode

try:
    from tiledb.cloud import groups
except ImportError:
    pass

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
        register_kwargs: Mapping[str, Any] = {},
        reader_kwargs: Mapping[str, Any] = {},
        compressor: tiledb.Filter = tiledb.ZstdFilter(level=0),
        generate_pyramid: bool = False,
        pyramid_mode: ScalerMode = ScalerMode.CHUNKED_PROGRESSIVE,
        pyramid_scale: List[int] = [],
        interpolation_method: ScaleMethod = ScaleMethod.NEAREST,
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
        :param chunked: If true, convert one tile at a time instead of the whole image.
            **Note**: The OpenSlideConverter may not be 100% lossless with chunked=True
            for levels>0, even though the converted images look visually identical to the
            original ones.
        :param max_workers: Maximum number of threads that can be used for conversion.
            Applicable only if chunked=True.
        :param register_kwargs: Cloud group registration optional args e.g namespace,
            parent_uri, storage_uri, credentials_name
        :param reader_kwargs: Keyword arguments passed to the _ImageReaderType constructor.
        :param generate_pyramid: If true, generate downsampled levels keeping only level 0 of the original image.
        :param pyramid_mode: The scaling approach used. Options other than the default may require more system memory.
        :param pyramid_scale: The downsample ratios of the generated levels.
        :param interpolation_method: The intrepolation method for downsampling the base image.

        :param compressor: TileDB compression filter
        """
        if cls._ImageReaderType is None:
            raise NotImplementedError(f"{cls} does not support importing")

        if tiledb.object_type(output_path) != "group":
            tiledb.group_create(output_path)

        pixel_depth = get_pixel_depth(compressor)
        max_tiles = cls._DEFAULT_TILES.copy()
        max_tiles.update(tiles)
        max_tiles["X"] *= pixel_depth

        with cls._ImageReaderType(input_path, **reader_kwargs) as reader:
            source_axes = reader.axes
            # Create a TileDB array for each level in range(level_min, reader.level_count)
            uris = []
            level_max = reader.level_count if not generate_pyramid else level_min + 1
            if reader.level_count > 1 and generate_pyramid:
                print(
                    "Warning: The image contains multiple levels but pyramid generation is enabled. All levels "
                    "except level zero will be skipped"
                )

            for level in range(level_min, level_max):
            levels_meta = []
            for level in range(level_min, reader.level_count):
                uri = os.path.join(output_path, f"l_{level}.tdb")
                if tiledb.object_type(uri) == "array":
                    # level has already been converted
                    continue

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

                # create TileDB array
                dim_shape = axes_mapper.map_shape(source_shape)
                attr_dtype = reader.level_dtype(level)
                schema = _get_schema(
                    dim_names, dim_shape, max_tiles, attr_dtype, compressor
                )
                tiledb.Array.create(uri, schema)
                uris.append(uri)

                # Calculate downsample factor
                dims_map = dict(zip(dim_names, dim_shape))
                if level == level_min:
                    l0_w, l0_h = dims_map["X"], dims_map["Y"]
                downsample_factor = (l0_w / dims_map["X"] + l0_h / dims_map["Y"]) / 2.0

                # Store layer mapping with shape value
                meta_kvstore = {
                    "uri": uri,
                    "level": level,
                    "axes": "".join(dim_names),
                    "shape": dim_shape,
                    "downsample_factor": downsample_factor,
                }
                levels_meta.append(meta_kvstore)

                # write image and metadata to TileDB array
                with tiledb.open(uri, "w") as a:
                    a.meta.update(reader.level_metadata(level), level=level)
                    if chunked or max_workers:
                        inv_axes_mapper = axes_mapper.inverted

                        def tile_to_tiledb(level_tile: Tuple[slice, ...]) -> None:
                            source_tile = inv_axes_mapper.map_tile(level_tile)
                            image = reader.level_image(level, source_tile)
                            a[level_tile] = axes_mapper.map_array(image)

                        ex = ThreadPoolExecutor(max_workers) if max_workers else None
                        mapper = getattr(ex, "map", map)
                        for _ in tqdm(
                            mapper(tile_to_tiledb, iter_tiles(a.domain)),
                            desc=f"Ingesting level {level}",
                            total=num_tiles(a.domain),
                            unit="tiles",
                        ):
                            pass
                        if ex:
                            ex.shutdown()
                    else:
                        image = reader.level_image(level)
                        a[:] = axes_mapper.map_array(image)
                        a[:] = axes_mapper.map_array(reader.level_image(level))

                    if generate_pyramid:
                        scaler = Scaler(
                            a,
                            Axes("".join(dim.name for dim in a.domain)),
                            scale_factor=pyramid_scale,
                        )
                uris.append(uri)

            if generate_pyramid:
                level = level_min + 1

                for dimension in scaler.resolutions:
                    uri = os.path.join(output_path, f"l_{level}.tdb")
                    schema = _get_schema(
                        axes=level_axes,
                        shape=dimension,
                        attr_dtype=level_dtype,
                        max_tiles=ChainMap(dict(tiles or {}), cls._DEFAULT_TILES),
                    )
                    tiledb.Array.create(uri, schema)
                    if (
                        pyramid_mode == ScalerMode.NON_PROGRESSIVE
                        or pyramid_mode == ScalerMode.CHUNKED_NON_PROGRESSIVE
                    ):
                        with tiledb.open(uris[0], "r") as r:
                            with tiledb.open(uri, "w") as a:
                                a.meta.update(metadata, level=level)
                                if pyramid_mode == ScalerMode.NON_PROGRESSIVE:
                                    scaler.apply(
                                        r,
                                        a,
                                        level - level_min - 1,
                                        interpolation_method,
                                    )
                                else:
                                    scaler.apply_chunked(
                                        r,
                                        a,
                                        level - level_min - 1,
                                        interpolation_method,
                                    )
                    else:
                        with tiledb.open(uris[-1], "r") as r:
                            with tiledb.open(uri, "w") as a:
                                a.meta.update(metadata, level=level)
                                if pyramid_mode == ScalerMode.PROGRESSIVE:
                                    scaler.apply_progressive(r, a, interpolation_method)
                                else:
                                    scaler.apply_chunked_progressive(
                                        r, a, interpolation_method
                                    )
                    uris.append(uri)
                    level += 1

            # Write group metadata
            with tiledb.Group(output_path, "w") as group:
                group.meta.update(
                    reader.group_metadata,
                    axes=source_axes.dims,
                    pkg_version=version,
                    fmt_version=FMT_VERSION,
                    dataset_type=DATASET_TYPE,
                    levels=json.dumps(levels_meta),
                )
                for uri in uris:
                    if urlparse(uri).scheme == "tiledb":
                        group.add(uri, relative=False)
                    else:
                        group.add(os.path.basename(uri), relative=True)

        # Register group in cloud if package exists
        if output_path.startswith("tiledb://"):
            groups.register(name=os.path.basename(output_path), **register_kwargs)


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
