from __future__ import annotations

import json
import logging
import os
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import jsonpickle
import numpy as np
from numpy._typing import NDArray
from tqdm import tqdm
from typing_extensions import Self

from .scale import Scaler

try:
    from tiledb.cloud.groups import register as register_group
except ImportError:
    register_group = None

import tiledb
from tiledb.filter import WebpFilter

from .. import ATTR_NAME
from ..helpers import (
    MetadataCallbackError,
    ReadWriteGroup,
    compute_channel_minmax,
    get_axes_mapper,
    get_axes_translation,
    get_logger_wrapper,
    get_schema,
    is_local_path,
    iter_levels_meta,
    iter_pixel_depths_meta,
    open_bioimg,
    remove_ome_image_metadata,
    resolve_path,
    validate_ingestion,
)
from ..openslide import TileDBOpenSlide
from ..version import version as PKG_VERSION
from . import DATASET_TYPE, DEFAULT_SCRATCH_SPACE, FMT_VERSION
from .axes import Axes
from .tiles import iter_tiles, num_tiles

# Define covariant type variables
TReader = TypeVar("TReader", bound="ImageReader")
TWriter = TypeVar("TWriter", bound="ImageWriter")


class ImageReader(Protocol):
    _logger: logging.Logger

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

    @property
    def source_ctx(self) -> tiledb.Ctx:
        """The ctx of the source path of this image reader."""
        ...

    @property
    def dest_ctx(self) -> tiledb.Ctx:
        """The ctx of the dest path of this image reader."""
        ...

    @property
    def logger(self) -> Optional[logging.Logger]:
        """The logger of this image reader."""
        ...

    @property
    def axes(self) -> Axes:
        """The axes of this multi-resolution image."""
        ...

    @property
    def channels(self) -> Sequence[str]:
        """Names of the channels (C axis) of this multi-resolution image."""
        ...

    @property
    def webp_format(self) -> WebpFilter.WebpInputFormat:
        """WebpInputFormat of this multi-resolution image. Defaults to WEBP_NONE."""
        return WebpFilter.WebpInputFormat.WEBP_NONE

    @property
    def level_count(self) -> int:
        """
        The number of levels for this multi-resolution image.

        Levels are numbered from 0 (highest resolution) to level_count - 1 (lowest resolution).
        """
        ...

    def level_dtype(self, level: int) -> np.dtype:
        """Return the dtype of the image for the given level."""
        ...

    def level_shape(self, level: int) -> Tuple[int, ...]:
        """Return the shape of the image for the given level."""
        ...

    def level_image(
        self, level: int, tile: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """
        Return the image for the given level as numpy array.

        The axes of the array are specified by the `axes` property.

        :param tile: If not None, a tuple of slices (one per each axes) that specify the
            subregion of the image to return.
        """
        ...

    def level_metadata(self, level: int) -> Dict[str, Any]:
        """Return the metadata for the given level."""
        ...

    @property
    def group_metadata(self) -> Dict[str, Any]:
        """Return the metadata for the whole multi-resolution image."""
        ...

    @property
    def image_metadata(self) -> Dict[str, Any]:
        """Return the metadata for the whole multi-resolution image."""
        ...

    @property
    def original_metadata(self) -> Dict[str, Any]:
        """Return the metadata of the original file."""
        ...

    def optimal_reader(
        self, level: int, max_workers: int = 0
    ) -> Union[None, Iterator[Tuple[Tuple[slice, ...], NDArray[Any]]]]:
        """
        Return an image tile iterator with optimal memory access pattern.

        :param level: The overview to read from
        :param max_workers: The number of thread to spawn to read from the file
        """
        ...


class ImageWriter(Protocol):

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

    def write_group_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Write metadata for the whole multi-resolution image."""
        ...

    def compute_level_metadata(
        self,
        baseline: bool,
        num_levels: int,
        image_dtype: np.dtype,
        group_metadata: Mapping[str, Any],
        array_metadata: Mapping[str, Any],
        **writer_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Compute the necessary metadata for the current level

        :param baseline: Sets current image as the baseline for the pyramid
        :param image_dtype: THe data type of the image to be stored
        :param group_metadata: The TileDB group pyramid metadata
        :param array_metadata: The TileDB array level metadata
        """
        ...

    def write_level_image(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> None:
        """
        Write the image for the given level.

        :param baseline: Sets current image as the baseline for the pyramid
        :param num_levels: The total number of reduced resolution images
        :param image: Image for the given level as numpy array
        :param metadata: Metadata for the given level
        :param image_mask: Mask the original image depending on export format requirements
        """
        ...


class ImageConverter(Protocol[TReader, TWriter]):
    _ImageReaderType: Optional[Type[TReader]] = None
    _ImageWriterType: Optional[Type[TWriter]] = None

    @classmethod
    def from_tiledb(cls, input_path: str, output_path: str) -> Type[Self]: ...

    @classmethod
    def to_tiledb(cls, source: str, output_path: str) -> Type[Self]: ...


class ImageConverterMixin(Generic[TReader, TWriter]):
    # setting a tile to "infinite" effectively makes it equal to the dimension size
    _DEFAULT_TILES = {"T": 1, "C": np.inf, "Z": 1, "Y": 1024, "X": 1024}
    _ImageReaderType: Optional[Type[TReader]] = None
    _ImageWriterType: Optional[Type[TWriter]] = None

    @classmethod
    def from_tiledb(
        cls,
        input_path: str,
        output_path: str,
        *,
        level_min: int = 0,
        attr: str = ATTR_NAME,
        config: Union[tiledb.Config, Mapping[str, Any]] = None,
        output_config: Union[tiledb.Config, Mapping[str, Any]] = None,
        scratch_space: str = DEFAULT_SCRATCH_SPACE,
        log: Optional[Union[bool, logging.Logger]] = None,
        **writer_kwargs: Mapping[str, Any],
    ) -> Type[ImageConverterMixin[TReader, TWriter]]:
        """
        Convert a TileDB Group of Arrays back to other format images, one per level
        :param input_path: path to the TileDB group of arrays
        :param output_path: path to the image
        :param level_min: minimum level of the image to be converted. By default set to 0
            to convert all levels
        :param attr: attribute name for backwards compatibility support
        :param config: tiledb configuration either a dict or a tiledb.Config of source
        :param output_config: tiledb configuration either a dict or a tiledb.Config of destination
        :param scratch_space: shared memory or cache space for cloud random access export support
        :param log: verbose logging, defaults to None. Allows passing custom logging.Logger or boolean.
        If None or bool=False it initiates an INFO level logging. If bool=True then a logger is instantiated in
        DEBUG logging level.
        """

        if cls._ImageWriterType is None:
            raise NotImplementedError(f"{cls} does not support exporting")

        # Initializes the logger depending on the API path chosen
        if log:
            logger = get_logger_wrapper(log) if isinstance(log, bool) else log
        else:
            default_verbose = False
            logger = get_logger_wrapper(default_verbose)

        out_uri_res, scheme = resolve_path(output_path)
        logger.debug(f"Resolving output path {out_uri_res}, with scheme {scheme}")
        vfs_use = False if is_local_path(scheme) else True
        logger.debug(f"VFS is used: {vfs_use}")

        # OS specific
        destination_uri = (
            os.path.join(scratch_space, os.path.basename(out_uri_res))
            if vfs_use
            else out_uri_res
        )
        logger.debug(f"Scratch space temp destination uri: {destination_uri}")

        if not output_config and vfs_use:
            output_config = config

        slide = TileDBOpenSlide(input_path, attr=attr, config=config)
        writer = cls._ImageWriterType(destination_uri, logger, **(writer_kwargs or {}))

        with slide, writer:
            writer.write_group_metadata(slide.properties)
            group_metadata = jsonpickle.loads(slide.properties.get("metadata", "{}"))
            for idx, _ in enumerate(slide.levels):
                if idx < level_min:
                    continue
                level_image = slide.read_level(idx, to_original_axes=True)
                level_metadata = writer.compute_level_metadata(
                    idx == level_min,
                    len(slide.levels) - level_min,
                    level_image.dtype,
                    group_metadata,
                    slide.level_properties(idx),
                    **writer_kwargs,
                )
                writer.write_level_image(
                    level_image,
                    level_metadata,
                )

            if vfs_use:
                # Flush to remote
                with open(destination_uri, "rb") as data:
                    vfs = tiledb.VFS(config=output_config)
                    with vfs.open(output_path, "wb") as dest:
                        dest.write(data.read())
        return cls

    @classmethod
    def to_tiledb(
        cls,
        source: str,
        output_path: str,
        *,
        level_min: int = 0,
        tiles: Optional[Mapping[str, int]] = None,
        tile_scale: int = 1,
        preserve_axes: bool = False,
        chunked: bool = False,
        max_workers: int = 0,
        exclude_metadata: Union[bool, Callable[[str], str], None] = None,
        experimental_reader: bool = False,
        experimental_queue_limit: Tuple[int, int] = (10, 20),
        compressor: Optional[Union[Mapping[int, Any], Any]] = None,
        log: Optional[Union[bool, logging.Logger]] = None,
        reader_kwargs: Optional[MutableMapping[str, Any]] = None,
        pyramid_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Type[ImageConverterMixin[TReader, TWriter]]:
        """
                Convert an image to a TileDB Group of Arrays, one per level.

                :param source: path to the input image or ImageReader object
                :param output_path: path to the TileDB group of arrays
                :param level_min: minimum level of the image to be converted. By default set to 0
                    to convert all levels.
                :param tiles: A mapping from dimension name (one of 'T', 'C', 'Z', 'Y', 'X') to
                    the (maximum) tile for this dimension.
                :param tile_scale: The scaling factor applied to each tile during I/O.
                    Larger scale factors will result in less I/O operations.
                :param preserve_axes: If true, preserve the axes order of the original image.
                :param chunked: If true, convert one tile at a time instead of the whole image.
                    **Note**: The OpenSlideConverter may not be 100% lossless with chunked=True
                    for levels>0, even though the converted images look visually identical to the
                    original ones.
                :param max_workers: Maximum number of threads that can be used for conversion.
                    Applicable only if chunked=True.
                :param exclude_metadata: An optional argument that specifies how to transform the original metadata.
                    It can be one of the following:
                    *   A callable (function, method, etc.) that takes an OME-XML string and returns it as a string, while removing
                        some of the original metadata and excluding them from being ingested.
                    *   A boolean value:
                        *   ``True``: Indicates a specific built-in transformation should be applied. see: `remove_ome_image_metadata`
                        *   ``False``: Indicates no transformation should be applied.
                    *   ``None``: Indicates no transformation should be applied (same as ``False``).
                :param experimental_reader: If true, use the experimental tiff reader optimized for s3 reads.
                    Experimental feature, use with caution
                :param experimental_queue_limit: When using the experimental reader, define the minimum and maximum number of
                    pending tiles waiting to be written to TileDB.
                :param compressor: TileDB compression filter mapping for each level
                :param log: verbose logging, defaults to None. Allows passing custom logging.Logger or boolean.
                    If None or bool=False it initiates an INFO level logging. If bool=True then a logger is instantiated in
                    DEBUG logging level.
                :param reader_kwargs: Keyword arguments passed to the _ImageReaderType constructor. Allows passing configuration
                    parameters like tiledb.Config or/and tiledb.Ctx.
        See Also        :param pyramid_kwargs: Keyword arguments passed to the scaler constructor for
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

        if log:
            logger = get_logger_wrapper(log) if isinstance(log, bool) else log
        else:
            default_verbose = False
            logger = get_logger_wrapper(default_verbose)

        # Backwards compatibility config v0.2.13
        if reader_kwargs:
            common_cfg = reader_kwargs.get("config", None)
            if common_cfg:
                # Overwrite the source and destination configs with the common
                reader_kwargs["source_config"] = reader_kwargs["dest_config"] = (
                    common_cfg
                )

        reader = cls._ImageReaderType(
            source, logger, **reader_kwargs if reader_kwargs else {}
        )

        max_tiles = cls._DEFAULT_TILES.copy()
        logger.debug(f"Max tiles:{max_tiles}")
        if tiles:
            max_tiles.update(tiles)
        logger.debug(f"Updated max tiles:{max_tiles}")

        rw_group = ReadWriteGroup(output_path, ctx=reader.dest_ctx)
        metadata = {}
        original_metadata = {}

        # Checks group validity
        overwrite = False
        with rw_group:
            if not rw_group.valid_group:
                logger.debug(
                    f"Already existing on disk Group at {output_path} evaluated as corrupted."
                )
                # Overwrite destination group
                rw_group.m_group.delete(recursive=True)
                overwrite = True
            else:
                # If group is valid we remove the KV and we will re-introduce it
                # after the current ingestion process as the final step
                del rw_group.w_group.meta["valid"]

        if overwrite:
            # Re-initializes the destination Group for re-ingestion
            logger.debug(f"Group {output_path} will be overwritten.")
            rw_group = ReadWriteGroup(output_path, ctx=reader.dest_ctx)

        with rw_group, reader:
            # Update MIME type
            rw_group.w_group.meta.update(dataset_type=DATASET_TYPE)
            stored_fmt_version = rw_group.r_group.meta.get("fmt_version")
            logger.debug(f"Format version: {stored_fmt_version}")
            if stored_fmt_version not in (None, FMT_VERSION):
                warnings.warn(
                    "Incremental ingestion is not supported for different versions: "
                    f"current version is {FMT_VERSION}, stored version is {stored_fmt_version} - "
                    f"Default Fallback: No changes will apply to already ingested image"
                )
                return cls

            # Check if compressor Mapping has 1-1 correspondance
            if isinstance(compressor, Mapping):
                if len(compressor.items()) != reader.level_count:
                    raise ValueError(
                        f"Compressor filter mapping does not map every level to a Filter {len(compressor.items())} != {reader.level_count}"
                    )

            compressors = {}
            for level in range(level_min, reader.level_count):
                if compressor is None:
                    compressors[level] = tiledb.ZstdFilter(level=0)
                elif isinstance(compressor, tiledb.Filter):
                    if (
                        isinstance(compressor, tiledb.WebpFilter)
                        and compressor.input_format
                        == WebpFilter.WebpInputFormat.WEBP_NONE
                    ):
                        compressor = tiledb.WebpFilter(
                            input_format=reader.webp_format,
                            quality=compressor.quality,
                            lossless=compressor.lossless,
                        )
                    # One filter is given apply to all levels
                    compressors[level] = compressor
                elif isinstance(compressor, Mapping):
                    compressors = compressor  # type: ignore
                    break

            logger.debug(f"Compressors : {compressors}")
            convert_kwargs = dict(
                reader=reader,
                rw_group=rw_group,
                max_tiles=max_tiles,
                tile_scale=tile_scale,
                preserve_axes=preserve_axes,
                chunked=chunked,
                max_workers=max_workers,
                compressor=compressors,
                experimental_reader=experimental_reader,
                experimental_queue_limits=experimental_queue_limit,
            )
            logger.debug(f"Convert arguments : {convert_kwargs}")

            metadata = reader.image_metadata
            metadata["axes"] = []
            channel_min_max = []
            scaled_compressors: Mapping[int, Any] = {}
            if pyramid_kwargs is not None:
                if level_min < reader.level_count - 1:
                    warnings.warn(
                        f"The image contains multiple levels but only level {level_min} "
                        "will be considered for generating the image pyramid"
                    )
                level_meta = _convert_level_to_tiledb(level_min, **convert_kwargs)

                scaled_compressors, levels_meta = _create_image_pyramid(
                    reader,
                    rw_group,
                    level_meta["uri"],
                    level_min,
                    max_tiles,
                    compressors,
                    preserve_axes,
                    pyramid_kwargs,
                )

                channel_min_max.append(level_meta["channelMinMax"])
                metadata["axes"] += levels_meta["axes"]
            else:
                for level in range(level_min, reader.level_count):
                    logger.info(f"Converting level: {level}")
                    level_meta = _convert_level_to_tiledb(level, **convert_kwargs)
                    channel_min_max.append(level_meta["channelMinMax"])
                    logger.debug(
                        f'Level {level} channel MinMax: {level_meta["channelMinMax"]}'
                    )
                    metadata["axes"].append(level_meta["axes"])
                    logger.debug(f'Level {level} axes: {level_meta["axes"]}')
            for idx in range(len(metadata["channels"])):
                metadata["channels"][idx].setdefault(
                    "min", channel_min_max[0].item((idx, 0))
                )
                metadata["channels"][idx].setdefault(
                    "max", channel_min_max[0].item((idx, 1))
                )

            metadata["channels"] = {f"{ATTR_NAME}": metadata["channels"]}
            logger.debug(f'Metadata channels: {metadata["channels"]}')

            if not exclude_metadata:
                original_metadata = reader.original_metadata
            else:
                if ome_xml := reader.original_metadata.get("ome_metadata"):
                    if isinstance(exclude_metadata, bool):
                        pruned_metadata = remove_ome_image_metadata(ome_xml)
                    elif callable(exclude_metadata):
                        try:
                            pruned_metadata = exclude_metadata(ome_xml)
                        except Exception as exc:
                            raise MetadataCallbackError(str(exc))
                    else:
                        raise TypeError("exclude_metadata must be bool or callable")
                    original_metadata = (
                        {"ome_metadata": pruned_metadata} if pruned_metadata else {}
                    )

        with rw_group:
            rw_group.w_group.meta.update(
                reader.group_metadata,
                axes=reader.axes.dims,
                pixel_depth=jsonpickle.encode(
                    dict(iter_pixel_depths_meta({**compressors, **scaled_compressors})),
                    unpicklable=False,
                ),
                pkg_version=PKG_VERSION,
                fmt_version=FMT_VERSION,
                channels=json.dumps(reader.channels),
                levels=jsonpickle.encode(
                    sorted(
                        iter_levels_meta(rw_group.r_group, ctx=reader.dest_ctx),
                        key=itemgetter("level"),
                    ),
                    unpicklable=False,
                ),
                metadata=jsonpickle.encode(metadata, unpicklable=False),
                original_metadata=jsonpickle.encode(original_metadata),
            )

        # This is the last step of the ingestion unless the following metadata KV
        # is not stored then we consider the ingested asset as corrupted
        with rw_group:
            rw_group.w_group.meta.update(valid=True)
        return cls


def _convert_level_to_tiledb(
    level: int,
    *,
    reader: ImageReader,
    rw_group: ReadWriteGroup,
    max_tiles: MutableMapping[str, int],
    tile_scale: int,
    preserve_axes: bool,
    chunked: bool,
    max_workers: int,
    compressor: Mapping[int, tiledb.Filter],
    experimental_reader: bool,
    experimental_queue_limits: Tuple[int, int],
) -> Mapping[str, Any]:
    level_metadata: MutableMapping[str, Any] = {}

    # create mapper from source to target axes
    source_axes = reader.axes
    source_shape = reader.level_shape(level)

    axes_mapper, dim_names, max_tiles = get_axes_mapper(
        reader.axes,
        reader.level_shape(level),
        compressor,
        level,
        preserve_axes,
        max_tiles,
    )
    # create TileDB schema
    dim_shape = axes_mapper.map_shape(source_shape)
    attr_dtype = reader.level_dtype(level)
    schema = get_schema(
        dim_names,
        dim_shape,
        max_tiles,
        attr_dtype,
        compressor.get(level, tiledb.ZstdFilter(level=0)),
    )

    # We need to calculate the min-max values per channel
    # First find the indices of all axes except 'C' needed for numpy amin and amax
    min_max_indices = tuple(
        idx for idx, char in enumerate(source_axes.dims) if char != "C"
    )

    # Find the number of channels
    channel_index = source_axes.dims.find("C")
    channel_count = source_shape[channel_index] if channel_index > -1 else 1

    # Initialize a numpy 2D array to hold the min-max values per channel
    channel_min_max = np.empty((channel_count, 2))

    if np.issubdtype(reader.level_dtype(0), np.integer):
        min_value = np.iinfo(reader.level_dtype(level)).min
        max_value = np.iinfo(reader.level_dtype(level)).max
    else:
        min_value = np.finfo(reader.level_dtype(level)).min
        max_value = np.finfo(reader.level_dtype(level)).max
    channel_min_max[:, 0] = np.repeat(max_value, channel_count)
    channel_min_max[:, 1] = np.repeat(min_value, channel_count)

    level_metadata["axes"] = {
        "originalAxes": [*reader.axes.dims],
        "originalShape": reader.level_shape(level),
        "storedAxes": dim_names,
        "storedShape": dim_shape,
        "axesMapping": get_axes_translation(
            compressor.get(level, tiledb.ZstdFilter(level=0)), reader.axes.dims
        ),
    }

    # get or create TileDB array uri
    uri, created = rw_group.get_or_create(f"l_{level}.tdb", schema)

    if created or not validate_ingestion(uri, ctx=reader.dest_ctx):
        # write image and metadata to TileDB array
        with open_bioimg(uri, "w", ctx=reader.dest_ctx) as out_array:
            out_array.meta.update(reader.level_metadata(level), level=level)
            inv_axes_mapper = axes_mapper.inverse
            if chunked:
                ex = ThreadPoolExecutor(max_workers) if max_workers else None
                mapper = getattr(ex, "map", map)
                opt_reader = reader.optimal_reader(level=level, max_workers=max_workers)

                if experimental_reader and opt_reader is not None:

                    def tile_to_tiledb_exp(
                        tile: Tuple[Tuple[slice, ...], NDArray[Any]]
                    ) -> Tuple[np.ndarray, ...]:
                        idx, data = tile
                        array_tile = axes_mapper.map_tile(idx)
                        out_array[array_tile] = axes_mapper.map_array(data)

                        # return a tuple containing the min-max values of the tile
                        return np.amin(data, axis=min_max_indices), np.amax(
                            data, axis=min_max_indices
                        )

                    if ex:
                        should_fetch_next = threading.Event()
                        should_fetch_next.clear()

                        def process(fut: Future[Tuple[np.ndarray, ...]]) -> None:
                            t_min, t_max = fut.result()
                            compute_channel_minmax(channel_min_max, t_min, t_max)

                            if ex._work_queue.qsize() < experimental_queue_limits[0]:  # type: ignore
                                should_fetch_next.set()

                        for tile in opt_reader:
                            future = ex.submit(tile_to_tiledb_exp, tile)
                            future.add_done_callback(process)

                            if ex._work_queue.qsize() > experimental_queue_limits[1]:
                                should_fetch_next.clear()
                                should_fetch_next.wait()

                        ex.shutdown()
                    else:
                        for tile in opt_reader:
                            t_min, t_max = tile_to_tiledb_exp(tile)
                            compute_channel_minmax(channel_min_max, t_min, t_max)
                else:

                    def tile_to_tiledb(
                        level_tile: Tuple[slice, ...]
                    ) -> Tuple[np.ndarray, ...]:
                        source_tile = inv_axes_mapper.map_tile(level_tile)
                        image = reader.level_image(level, source_tile)
                        out_array[level_tile] = axes_mapper.map_array(image)

                        # return a tuple containing the min-max values of the tile
                        return np.amin(image, axis=min_max_indices), np.amax(
                            image, axis=min_max_indices
                        )

                    for tile_min, tile_max in tqdm(
                        mapper(tile_to_tiledb, iter_tiles(out_array.domain)),
                        desc=f"Ingesting level {level}",
                        total=num_tiles(out_array.domain),
                        unit="tiles",
                    ):
                        # Find the global min-max values from all tiles
                        compute_channel_minmax(channel_min_max, tile_min, tile_max)
                        pass
                    if ex:
                        ex.shutdown()
            else:
                image = reader.level_image(level)
                ex = ThreadPoolExecutor(max_workers) if max_workers else None
                mapper = getattr(ex, "map", map)

                def write_to_tiledb(level_tile: Tuple[slice, ...]) -> None:
                    source_tile = inv_axes_mapper.map_tile(level_tile)
                    out_array[level_tile] = axes_mapper.map_array(image[source_tile])

                for _ in tqdm(
                    mapper(
                        write_to_tiledb, iter_tiles(out_array.domain, scale=tile_scale)
                    ),
                    desc=f"Ingesting level {level}",
                    total=num_tiles(out_array.domain, scale=tile_scale),
                    unit="tiles",
                ):
                    # Find the global min-max values from all tiles
                    pass
                if ex:
                    ex.shutdown()

                compute_channel_minmax(
                    channel_min_max,
                    np.amin(image, axis=min_max_indices),
                    np.amax(image, axis=min_max_indices),
                )

    level_metadata["uri"] = uri
    level_metadata["channelMinMax"] = channel_min_max

    return level_metadata


def _create_image_pyramid(
    reader: ImageReader,
    rw_group: ReadWriteGroup,
    base_uri: str,
    base_level: int,
    max_tiles: MutableMapping[str, int],
    compressors: Mapping[int, tiledb.Filter],
    preserve_axes: bool,
    pyramid_kwargs: Mapping[str, Any],
) -> Tuple[Mapping[int, tiledb.Filter], Mapping[str, Any]]:
    scaler = Scaler(reader.level_shape(base_level), reader.axes.dims, **pyramid_kwargs)

    levels_metadata: MutableMapping[str, Any] = {"axes": []}

    for i, dim_shape in enumerate(scaler.level_shapes):
        level = base_level + 1 + i

        # The compressor of level 0 is used for the newly created scaled levels
        scaler.update_compressors(level, compressors[base_level])
        axes_mapper, dim_names, max_tiles = get_axes_mapper(
            reader.axes, dim_shape, scaler.compressors, level, preserve_axes, max_tiles
        )

        schema = get_schema(
            dim_names,
            axes_mapper.map_shape(dim_shape),
            max_tiles,
            reader.level_dtype(base_level),
            compressors[base_level],
        )
        uri, created = rw_group.get_or_create(f"l_{level}.tdb", schema)
        if not created:
            continue

        levels_metadata["axes"].append(
            {
                "originalAxes": [*reader.axes.dims],
                "originalShape": dim_shape,
                "storedAxes": dim_names,
                "storedShape": axes_mapper.map_shape(dim_shape),
                "axesMapping": get_axes_translation(
                    scaler.compressors[level], reader.axes.dims
                ),
            }
        )

        with open_bioimg(uri, mode="w") as out_array:
            out_array.meta.update(level=level)
            with open_bioimg(base_uri) as in_array:
                scaler.apply(in_array, out_array, i, axes_mapper)

        # if a non-progressive method is used, the input layer of the scaler
        # is the base image layer else we use the previously generated layer
        if scaler.progressive:
            base_uri = uri

    return scaler.compressors, levels_metadata
