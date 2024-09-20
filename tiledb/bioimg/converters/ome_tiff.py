from __future__ import annotations

import decimal
import logging
import math
import warnings
from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import jsonpickle as json
import numpy as np
from numpy._typing import NDArray

try:
    import tifffile
except ImportError as err:
    warnings.warn(
        "OMETiff Converter requires 'tifffile' package. "
        "You can install 'tiledb-bioimg' with the 'tiff' or 'full' flag"
    )
    raise err

from tiledb import VFS, Config, Ctx
from tiledb.cc import WebpInputFormat
from tiledb.highlevel import _get_ctx

from .. import ATTR_NAME, EXPORT_TILE_SIZE, WHITE_RGBA
from ..helpers import (
    get_decimal_from_rgba,
    get_logger_wrapper,
    get_rgba,
    iter_color,
)
from .axes import Axes
from .base import ImageConverterMixin
from .io import as_array
from .metadata import qpi_image_meta, qpi_original_meta


class OMETiffReader:

    _logger: logging.Logger

    def __init__(
        self,
        input_path: str,
        logger: Optional[logging.Logger] = None,
        *,
        source_config: Optional[Config] = None,
        source_ctx: Optional[Ctx] = None,
        dest_config: Optional[Config] = None,
        dest_ctx: Optional[Ctx] = None,
        extra_tags: Sequence[Union[str, int]] = (),
        buffer_size: Optional[int] = None,
    ):
        """
        OME-TIFF image reader

        :param input_path: The path to the TIFF image
        :param extra_tags: Extra tags to read, specified either by name or by int code.
        """
        self._logger = get_logger_wrapper(False) if not logger else logger
        self._extra_tags = extra_tags
        self._buffer_size = buffer_size

        # Use VFS for all paths local or remote for reading the input image
        self._input_path = input_path
        self._source_ctx = _get_ctx(source_ctx, source_config)
        self._source_cfg = self._source_ctx.config()
        self._dest_ctx = _get_ctx(dest_ctx, dest_config)
        self._dest_cfg = self._dest_ctx.config()
        self._vfs = VFS(config=self._source_cfg, ctx=self._source_ctx)
        self._vfs_fh = self._vfs.open(input_path, mode="rb")
        self._tiff = tifffile.TiffFile(self._vfs_fh)
        # XXX ignore all but the first series
        self._series = self._tiff.series[0]
        omexml = self._tiff.ome_metadata
        self._metadata = tifffile.xml2dict(omexml) if omexml else {}

    def __enter__(self) -> OMETiffReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._tiff.close()
        self._vfs.close(file=self._vfs_fh)

    @property
    def source_ctx(self) -> Ctx:
        return self._source_ctx

    @property
    def dest_ctx(self) -> Ctx:
        return self._dest_ctx

    @property
    def logger(self) -> Optional[logging.Logger]:
        return self._logger

    @property
    def axes(self) -> Axes:
        axes = Axes(self._series.axes.replace("S", "C"))
        self._logger.debug(f"Reader axes: {axes}")
        return axes

    @property
    def channels(self) -> Sequence[str]:
        # channel names are fixed if this is an RGB image
        if self.webp_format is WebpInputFormat.WEBP_RGB:
            self._logger.debug(f"Webp format: {WebpInputFormat.WEBP_RGB}")
            return "RED", "GREEN", "BLUE"

        # otherwise try to infer them from the OME-XML metadata
        self._logger.debug(f"Webp format is not: {WebpInputFormat.WEBP_RGB}")
        try:
            channels = self._metadata["OME"]["Image"][0]["Pixels"]["Channel"]
            if not isinstance(channels, Sequence):
                channels = [channels]
            self._logger.debug(f"Channels inferred: {channels}")
        except KeyError:
            return ()
        return tuple(c.get("Name") or f"Channel {i}" for i, c in enumerate(channels))

    @property
    def webp_format(self) -> WebpInputFormat:
        self._logger.debug(f"Keyframe photometric: {self._series.keyframe.photometric}")
        if self._series.keyframe.photometric == tifffile.PHOTOMETRIC.RGB:
            return WebpInputFormat.WEBP_RGB
        # XXX: it is possible that instead of a single RGB channel (samplesperpixel==3)
        # there are 3 MINISBLACK channels (samplesperpixel=1). In this case look for the
        # photometric interpretation in the original metadata
        if self._original_metadata("PhotometricInterpretation") == "RGB":
            return WebpInputFormat.WEBP_RGB
        return WebpInputFormat.WEBP_NONE

    @property
    def level_count(self) -> int:
        level_count = len(self._series.levels)
        self._logger.debug(f"Level count: {level_count}")
        return level_count

    def level_dtype(self, level: int) -> np.dtype:
        dtype = self._series.levels[level].dtype
        self._logger.debug(f"Level {level} dtype: {dtype}")
        return dtype

    def level_shape(self, level: int) -> Tuple[int, ...]:
        l_shape = cast(Tuple[int, ...], self._series.levels[level].shape)
        self._logger.debug(f"Level {level} shape: {l_shape}")
        return l_shape

    def level_image(
        self, level: int, tile: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        if tile is None:
            return self._series.levels[level].asarray()
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr required for reading a Tiff tile region")
        if not hasattr(self, "_zarr_group"):
            store = self._series.aszarr(multiscales=True)
            self._zarr_group = zarr.open(store, mode="r")
        return np.asarray(self._zarr_group[level][tile])

    def level_metadata(self, level: int) -> Dict[str, Any]:
        if level == 0:
            metadata = dict(self._metadata, axes=self._series.axes)
        else:
            metadata = None
        self._logger.debug(f"Level {level} - Metadata: {metadata}")
        keyframe = self._series.levels[level].keyframe
        self._logger.debug(f"Level {level} -Keyframe: {keyframe}")
        extratags = []
        get_tag = keyframe.tags.get
        self._logger.debug(f"Level {level} - Tag: {keyframe}")
        for key in self._extra_tags:
            tag = get_tag(key)
            if tag is not None:
                extratags.append(tag.astuple())
        self._logger.debug(f"Level {level} - Extratags: {extratags}")
        write_kwargs = dict(
            subifds=self.level_count - 1 if level == 0 else None,
            metadata=metadata,
            extratags=extratags,
            photometric=keyframe.photometric,
            planarconfig=keyframe.planarconfig,
            extrasamples=keyframe.extrasamples,
            rowsperstrip=keyframe.rowsperstrip,
            bitspersample=keyframe.bitspersample,
            compression=keyframe.compression,
            predictor=keyframe.predictor,
            subsampling=keyframe.subsampling,
            jpegtables=keyframe.jpegtables,
            colormap=keyframe.colormap,
            subfiletype=keyframe.subfiletype,
            software=keyframe.software,
            tile=keyframe.tile,
            datetime=keyframe.datetime,
            resolution=keyframe.resolution,
            resolutionunit=keyframe.resolutionunit,
        )
        return {"json_write_kwargs": json.dumps(write_kwargs)}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        writer_kwargs = dict(
            bigtiff=self._tiff.is_bigtiff,
            byteorder=self._tiff.byteorder,
            append=self._tiff.is_appendable,
            imagej=self._tiff.is_imagej,
            ome=self._tiff.is_ome,
        )
        self._logger.debug(f"Group metadata: {writer_kwargs}")
        return {"json_tiffwriter_kwargs": json.dumps(writer_kwargs)}

    def _original_metadata(self, key: str, default: Any = None) -> Any:
        try:
            xmlanns = self._metadata["OME"]["StructuredAnnotations"]["XMLAnnotation"]
            for xmlann in xmlanns:
                entry = xmlann["Value"]["OriginalMetadata"]
                if entry["Key"] == key:
                    return entry["Value"]
        except KeyError:
            pass
        return default

    @property
    def image_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        if self._metadata and self._tiff.is_ome:
            image_meta = self._metadata["OME"]["Image"]
            image = (
                image_meta[0]["Pixels"]
                if isinstance(image_meta, list)
                else image_meta["Pixels"]
            )

            channels = []
            image_channels = (
                image["Channel"]
                if isinstance(image["Channel"], list)
                else [image["Channel"]]
            )
            color_generator = iter_color(np.dtype(np.uint8), len(image_channels))

            for idx, channel in enumerate(image_channels):
                if "SamplesPerPixel" in channel and channel["SamplesPerPixel"] != 1:
                    color_generator = iter_color(
                        np.dtype(np.uint8), channel["SamplesPerPixel"]
                    )
                    for i in range(channel["SamplesPerPixel"]):
                        channel_metadata = {
                            "id": f'{channel.get("ID", f"{idx}")}-{i}',
                            "name": channel.get("Name", f"Channel {idx}-{i}"),
                            "color": next(color_generator),
                        }

                        channels.append(channel_metadata)
                else:
                    channel_metadata = {
                        "id": channel.get("ID", f"{idx}"),
                        "name": channel.get("Name", f"Channel {idx}"),
                        "color": (
                            get_rgba(channel["Color"])
                            if "Color" in channel
                            else next(color_generator)
                        ),
                    }
                    if "EmissionWavelength" in channel:
                        channel_metadata["emissionWavelength"] = channel[
                            "EmissionWavelength"
                        ]
                        channel_metadata["emissionWavelengthUnit"] = channel.get(
                            "EmissionWavelengthUnit", "nm"
                        )

                    channels.append(channel_metadata)

            metadata["channels"] = channels

            for dim in ["X", "Y", "Z"]:
                if f"PhysicalSize{dim}" in image:
                    metadata[f"physicalSize{dim}"] = image[f"PhysicalSize{dim}"]
                    metadata[f"physicalSize{dim}Unit"] = image.get(
                        f"PhysicalSize{dim}Unit", "µm"
                    )

            if "TimeIncrement" in image:
                metadata["timeIncrement"] = image["TimeIncrement"]
                metadata["timeIncrementUnit"] = image.get("TimeIncrementUnit", "s")
        elif self._tiff.is_qpi:
            metadata = qpi_image_meta(self._tiff.series[0])
        else:
            # If file is not OME we will try to extract metadata from the IFD.
            # If you are ingesting a non-OME tiff file you may need to provide a custom metadata
            # extraction method

            page = self._tiff.pages.first

            if page.photometric == tifffile.PHOTOMETRIC.RGB:
                color_generator = iter_color(np.dtype(np.uint8), 3)
                metadata["channels"] = [
                    {"id": f"{idx}", "name": f"{name}", "color": next(color_generator)}
                    for idx, name in enumerate(["red", "green", "blue"])
                ]
            else:
                num_channels, color_generator = (
                    (
                        self._series.shape[self.axes.dims.index("C")],
                        iter_color(
                            np.dtype(np.uint8),
                            self._series.shape[self.axes.dims.index("C")],
                        ),
                    )
                    if "C" in self.axes.dims
                    else (1, iter_color(np.dtype(np.uint8), 1))
                )

                metadata["channels"] = [
                    {
                        "id": f"{idx}",
                        "name": f"Channel {idx}",
                        "color": next(
                            color_generator, get_rgba(WHITE_RGBA)
                        ),  # decimal representation of white
                    }
                    for idx in range(num_channels)
                ]

            # Aperio .svs files have a description tag with metadata split by '|' https://openslide.org/formats/aperio/
            if self._tiff.is_svs:
                info = {}
                for entry in page.description.split("\n")[1].split("|"):
                    key, value = entry.split("=")
                    info[key] = value

                if "MPP" in info:
                    metadata["physicalSizeX"] = metadata["physicalSizeY"] = float(
                        info.get("MPP", "1")
                    )
                    metadata["physicalSizeXUnit"] = metadata["physicalSizeYUnit"] = "µm"

        self._logger.debug(f"Image metadata: {metadata}")
        return metadata

    @property
    def original_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        for attr in dir(self._tiff):
            try:
                value = getattr(self._tiff, attr)
                if attr.endswith("metadata") and value:
                    metadata.setdefault(attr, value)
            except IndexError:
                warnings.warn(f"Failed to read {attr}")

        if self._tiff.is_svs:
            metadata.setdefault("svs_metadata", self._tiff.pages.first.description)
        elif self._tiff.is_qpi:
            metadata.setdefault("qpi_metadata", qpi_original_meta(self._tiff))

        return metadata

    def optimal_reader(
        self, level: int, max_workers: int = 0
    ) -> Optional[Iterator[Tuple[Tuple[slice, ...], NDArray[Any]]]]:
        # Get the pages the hold the data for the requested level
        pages = self._series.levels[level].pages
        extra_dims = pages.shape[: len(pages.axes) - len(pages[0].axes)] + (1,)

        # construct a generator function to read the image in optimal order
        def chunk_iterator() -> Iterator[Tuple[Tuple[slice, ...], NDArray[Any]]]:
            for idx, page in enumerate(pages):
                for data, offset in as_array(
                    page, logger=get_logger_wrapper(), buffer_size=self._buffer_size
                ):
                    extra_offsets: Tuple[int, ...] = ()
                    for i in range(len(extra_dims) - 1):
                        dim_index = (
                            idx // math.prod(extra_dims[i + 1 :])
                        ) % extra_dims[i]
                        extra_offsets = extra_offsets + (dim_index,)
                        data.shape = (1,) + data.shape

                    offset = extra_offsets + offset

                    slices: Tuple[slice, ...] = ()
                    for i in range(len(offset)):
                        slices = slices + (slice(offset[i], offset[i] + data.shape[i]),)

                    yield slices, data

        return chunk_iterator()


class OMETiffWriter:
    def __init__(self, output_path: str, logger: logging.Logger, ome: bool = True):
        self._logger = logger
        self._output_path = output_path
        self._ome = ome

    def __enter__(self) -> OMETiffWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._writer.close()

    def write_group_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._writer = tifffile.TiffWriter(
            self._output_path, shaped=False, bigtiff=True, append=False, ome=self._ome
        )

    def compute_level_metadata(
        self,
        baseline: bool,
        num_levels: int,
        image_dtype: np.dtype,
        group_metadata: Mapping[str, Any],
        array_metadata: Mapping[str, Any],
        **writer_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        original_axes = "".join(group_metadata.get("axes", [])[0].get("originalAxes"))
        original_shape = group_metadata.get("axes", [])[0].get("originalShape")

        writer_metadata: Dict[str, Any] = {}

        if baseline:
            writer_metadata["subifds"] = num_levels - 1 if num_levels > 1 else None
        else:
            writer_metadata["subfiletype"] = tifffile.FILETYPE.REDUCEDIMAGE

        writer_metadata["tile"] = (EXPORT_TILE_SIZE, EXPORT_TILE_SIZE)

        if self._ome:
            channel_metadata = group_metadata.get("channels", {}).get(ATTR_NAME, [])

            writer_metadata["metadata"] = {
                "axes": original_axes,
                "Channel": {
                    "Name": [
                        channel.get("name", f"Channel {idx}")
                        for idx, channel in enumerate(channel_metadata)
                    ],
                    "Color": [
                        str(get_decimal_from_rgba(channel.get("color")))
                        for channel in channel_metadata
                    ],
                },
            }

            ctx = decimal.Context()
            ctx.prec = 32

            for dim in ["X", "Y", "Z"]:
                if dim in original_axes:
                    writer_metadata["metadata"][f"PhysicalSize{dim}"] = format(
                        ctx.create_decimal(
                            repr(group_metadata.get(f"physicalSize{dim}", 1))
                        ),
                        "f",
                    )
                    writer_metadata["metadata"][f"PhysicalSize{dim}Unit"] = (
                        group_metadata.get(f"physicalSize{dim}Unit", "µm")
                    )

            if "T" in original_axes:
                writer_metadata["metadata"]["TimeIncrement"] = format(
                    ctx.create_decimal(repr(group_metadata.get("timeIncrement", 1))),
                    "f",
                )
                writer_metadata["metadata"]["TimeIncrementUnit"] = group_metadata.get(
                    "timeIncrementUnit", "s"
                )

            writer_metadata.update(writer_kwargs.get("tiff_options", {}))

            if (
                "C" in original_axes
                and original_shape[original_axes.index("C")] == 3
                and image_dtype == np.dtype(np.uint8)
            ):
                writer_metadata.setdefault("compression", tifffile.COMPRESSION.JPEG)
                writer_metadata.setdefault("compressionargs", {"level": 70})
                writer_metadata.setdefault("photometric", tifffile.PHOTOMETRIC.RGB)
            else:
                writer_metadata.setdefault(
                    "compression", tifffile.COMPRESSION.ADOBE_DEFLATE
                )
                writer_metadata.setdefault(
                    "photometric",
                    json.loads(array_metadata.get("json_write_kwargs", "{}")).get(
                        "photometric", tifffile.PHOTOMETRIC.MINISBLACK
                    ),
                )
        self._logger.debug(f"Writer metadata: {writer_metadata}")
        return writer_metadata

    def write_level_image(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> None:
        write_kwargs: Dict[str, Any] = dict(metadata)
        self._writer.write(image, **write_kwargs)


class OMETiffConverter(ImageConverterMixin[OMETiffReader, OMETiffWriter]):
    """Converter of Tiff-supported images to TileDB Groups of Arrays"""

    _ImageReaderType = OMETiffReader
    _ImageWriterType = OMETiffWriter
