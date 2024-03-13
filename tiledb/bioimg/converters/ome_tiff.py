import decimal
import logging
import math
import warnings
from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import jsonpickle as json
import numpy as np
import tifffile

from tiledb.bioimg.converters.io import asarray
from tiledb.cc import WebpInputFormat

from .. import ATTR_NAME, EXPORT_TILE_SIZE, WHITE_RGBA
from ..helpers import get_decimal_from_rgba, get_logger_wrapper, get_rgba, iter_color
from .axes import Axes
from .base import ImageConverter, ImageReader, ImageWriter
from .metadata import qpi_image_meta, qpi_original_meta


class OMETiffReader(ImageReader):
    def __init__(
        self,
        input_path: str,
        logger: Optional[logging.Logger] = None,
        extra_tags: Sequence[Union[str, int]] = (),
    ):
        """
        OME-TIFF image reader

        :param input_path: The path to the TIFF image
        :param extra_tags: Extra tags to read, specified either by name or by int code.
        """
        self._logger = get_logger_wrapper(False) if not logger else logger
        self._extra_tags = extra_tags
        self._tiff = tifffile.TiffFile(input_path)
        # XXX ignore all but the first series
        self._series = self._tiff.series[0]
        omexml = self._tiff.ome_metadata
        self._metadata = tifffile.xml2dict(omexml) if omexml else {}

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._tiff.close()

    @property
    def logger(self) -> Optional[logging.Logger]:
        return self._logger

    @logger.setter
    def logger(self, default_logger: logging.Logger) -> None:
        self._logger = default_logger

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
                        "color": get_rgba(channel["Color"])
                        if "Color" in channel
                        else next(color_generator),
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

    @property
    def page_tile_store_order(self) -> Mapping[int, int]:
        tile_store_order: MutableMapping[int, int] = {}

        for level in range(self.level_count):
            # If the page is of instance TiffFrame the hash is the same for all frames of a page
            for page in self._series.levels[level].pages:
                if page.axes.startswith("S"):
                    tile_store_order.setdefault(page.hash, 0)
                    continue

                z_length = page.chunked[0] if page.axes.startswith("Z") else 1
                y_length = (
                    page.chunked[1] if page.axes.startswith("Z") else page.chunked[0]
                )
                x_length = (
                    page.chunked[2] if page.axes.startswith("Z") else page.chunked[1]
                )

                # Row major detection loop
                pos = page.dataoffsets[0] + page.databytecounts[0]
                for z in range(z_length):
                    for y in range(y_length):
                        for x in range(x_length):
                            index = z * y_length * x_length + y * x_length + x

                            if index == 0:
                                continue

                            if page.dataoffsets[index] == pos:
                                pos += page.databytecounts[index]
                            else:
                                tile_store_order.setdefault(page.hash, 0)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    tile_store_order[page.hash] = 1
                    continue

                # Column major detection loop
                pos = page.dataoffsets[0] + page.databytecounts[0]
                for x in range(x_length):
                    for y in range(y_length):
                        for z in range(z_length):
                            index = z * y_length * x_length + y * x_length + x

                            if index == 0:
                                continue

                            if page.dataoffsets[index] == pos:
                                pos += page.databytecounts[index]
                            else:
                                tile_store_order.setdefault(page.hash, 0)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    tile_store_order[page.hash] = 2
        return tile_store_order

    def iter_mem_contig_tiles(
        self, level: int, chunk_target_size: int = 256
    ) -> Iterator[Tuple[slice, ...]]:
        pages = self._series.levels[level].pages

        if any(self.page_tile_store_order[page.hash] == 0 for page in pages):
            raise ValueError("Non memory contiguous images are not supported")

        dim_dif = len(self.axes.dims) - len(pages[0].axes)
        strides: MutableSequence[int] = [
            int(np.prod([d for d in self.level_shape(level)[:dim_dif]]))
        ]
        for dim in range(len(self.axes.dims) - len(pages[0].axes)):
            strides.append(
                int(np.prod([d for d in self.level_shape(level)[dim + 1 : dim_dif]]))
            )

        for idx in range(len(pages)):
            page = pages[idx]
            page_mask: Tuple[slice, ...] = ()
            samples_per_pixel: int = 1

            for dim in range(1, dim_dif + 1):
                dim_index = (idx % strides[dim - 1]) // strides[dim]
                page_mask += (slice(dim_index, dim_index + 1),)

            if page.axes == "YX" or page.axes == "ZYX":
                shape = [dim for dim in page.chunked]
                chunk_shape = [dim for dim in page.chunks]
                page_shape = [dim for dim in page.keyframe.shape]
            elif page.axes == "YXS" or page.axes == "ZYXS":
                shape = [dim for dim in page.chunked[:-1]]
                chunk_shape = [dim for dim in page.chunks[:-1]]
                page_shape = [dim for dim in page.keyframe.shape[:-1]]
                samples_per_pixel = page.keyframe.shape[-1]
            else:
                raise ValueError("Separate pixel sample images are not supported")

            if self.page_tile_store_order[page.hash] == 1:  # Row major
                # Calculate optimal tile scale for the requested page
                scale = int(
                    math.ceil(
                        chunk_target_size
                        / (
                            chunk_shape[0]
                            * np.prod(page_shape[1:])
                            * samples_per_pixel
                            / 2**20
                        )
                    )
                )
                if (
                    page.compression != 1
                ):  # if a compression scheme is applied, assume a 50% compression ratio
                    scale *= 2

                for row in range(0, shape[0], scale):
                    row_start = row * chunk_shape[0]
                    row_end = min((row + scale) * chunk_shape[0], page_shape[0])
                    mask = (slice(row_start, row_end),) + tuple(
                        slice(0, dim) for dim in page_shape[1:]
                    )

                    if page.axes.endswith("S"):
                        mask += (slice(0, page.samplesperpixel),)

                    yield page_mask + mask
            elif self.page_tile_store_order[page.hash] == 2:  # Column major
                scale = int(
                    math.ceil(
                        chunk_target_size
                        / (
                            chunk_shape[-1]
                            * np.prod(page_shape[:-1])
                            * samples_per_pixel
                            / 2**20
                        )
                    )
                )
                if (
                    page.compression != 1
                ):  # if a compression scheme is applied, assume a 50% compression ratio
                    scale *= 2

                for column in range(0, shape[-1], scale):
                    column_start = column * chunk_shape[-1]
                    column_end = min((column + scale) * chunk_shape[-1], page_shape[-1])
                    mask = tuple(slice(0, dim) for dim in page_shape[:-1]) + (
                        slice(column_start, column_end),
                    )
                    if page.axes.endswith("S"):
                        mask += (slice(0, page.samplesperpixel),)

                    yield page_mask + mask
            else:
                raise ValueError(
                    f"Unknown tile order value. Expected 0, 1 or 2, found {self.page_tile_store_order[page.hash]}"
                )

    def level_image_experimental(
        self, level: int, tile: Tuple[slice, ...]
    ) -> np.ndarray:
        pages = self._series.levels[level].pages
        dim_dif = len(self.axes.dims) - len(pages[0].axes)
        page_mask, mask = tile[:dim_dif], tile[dim_dif:]
        idx = page_mask[0].start if len(page_mask) else 0
        for dim, sl in enumerate(page_mask[1:]):
            idx = idx * self.level_shape(level)[dim + 1] + sl.start

        if idx >= len(pages):
            raise IndexError(
                f"Page index out of bounds. Expected less than {len(pages)} found {idx}"
            )

        page = pages[idx]
        page_axes = page.axes.replace("S", "C")

        axes_offsets: Tuple[int, int] = (
            self.axes.dims.index(page_axes[0]),
            self.axes.dims.index(page_axes[-1]),
        )

        if self.page_tile_store_order[page.hash] == 0:
            raise ValueError("Chunked mode is unsupported for random tile locations")

        # Remove unary dimensions from the chunks
        shape = [dim for dim in page.chunked if dim != 1]

        # the number of chunks per row
        size = int(np.prod(shape[1:]))

        if self.page_tile_store_order[page.hash] == 1:  # Row major
            row_width = page.chunks[0]
            row = int(math.floor(mask[0].start / row_width))
            scale_factor = int(math.ceil((mask[0].stop - mask[0].start) / row_width))

            result = asarray(
                page,
                slices=[slice(row * size, (row + scale_factor) * size, 1)],
                maxworkers=tifffile.TIFF.MAXWORKERS,
                storedTilesOrder=self.page_tile_store_order[page.hash],
            )

            result.shape = (
                tuple(1 for _ in range(0, axes_offsets[0]))
                + result.shape
                + tuple(1 for _ in range(axes_offsets[1] + 1, len(self.axes.dims)))
            )

            return result[
                tuple(slice(m.stop - m.start) if m.start else slice(None) for m in tile)
            ]
        elif self.page_tile_store_order[page.hash] == 2:  # Column major
            column_width = page.chunks[-2]
            column = int(math.floor(mask[-2].start / column_width))
            scale_factor = int(
                math.ceil((mask[-2].stop - mask[-2].start) / column_width)
            )

            slices = [
                slice(idx, None, size) for idx in range(column, column + scale_factor)
            ]
            result = asarray(
                page,
                slices=slices,
                maxworkers=tifffile.TIFF.MAXWORKERS,
                storedTilesOrder=self.page_tile_store_order[page.hash],
            )

            return result[
                tuple(slice(m.stop - m.start) if m.start else slice(None) for m in tile)
            ]


class OMETiffWriter(ImageWriter):
    def __init__(self, output_path: str, logger: logging.Logger, ome: bool = True):
        self._logger = logger
        self._output_path = output_path
        self._ome = ome

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
                    writer_metadata["metadata"][
                        f"PhysicalSize{dim}Unit"
                    ] = group_metadata.get(f"physicalSize{dim}Unit", "µm")

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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._writer.close()


class OMETiffConverter(ImageConverter):
    """Converter of Tiff-supported images to TileDB Groups of Arrays"""

    _ImageReaderType = OMETiffReader
    _ImageWriterType = OMETiffWriter
