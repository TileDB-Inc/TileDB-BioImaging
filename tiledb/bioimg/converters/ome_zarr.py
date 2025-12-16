from __future__ import annotations

import json
import logging
import os
import warnings
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy
import numpy as np
from numpy._typing import NDArray

try:
    import zarr
    from ome_zarr.format import Format, FormatV01, FormatV02, FormatV03, FormatV04
    from ome_zarr.reader import OMERO, Multiscales, Reader, ZarrLocation
    from ome_zarr.writer import write_multiscale
except ImportError as err:
    warnings.warn(
        "OMEZarr Converter requires 'ome-zarr' package. "
        "You can install 'tiledb-bioimg' with the 'zarr' or 'full' flag"
    )
    raise err

try:
    from ome_zarr.format import FormatV05

    HAS_FORMAT_V05 = True
except ImportError:
    warnings.warn(
        "FormatV05 not available. Zarr v3 format requires ome-zarr>=0.9.0. "
        "Falling back to FormatV04 (Zarr v2). "
        "To use Zarr v3, upgrade: pip install --upgrade ome-zarr",
        UserWarning,
        stacklevel=2,
    )
    FormatV05 = None
    HAS_FORMAT_V05 = False

if HAS_FORMAT_V05:
    from zarr.storage import FsspecStore, LocalStore
else:
    # Zarr v2 doesn't have these in zarr.storage
    try:
        from zarr.storage import DirectoryStore as LocalStore
        from zarr.storage import FSStore as FsspecStore
    except ImportError:
        # Fallback for older Zarr v2
        FsspecStore = None
        LocalStore = None

from tiledb import Config, Ctx
from tiledb.filter import WebpFilter
from tiledb.highlevel import _get_ctx

from .. import WHITE_RGB
from ..helpers import get_logger_wrapper, get_rgba, translate_config_to_s3fs
from .axes import Axes
from .base import ImageConverterMixin


def encode_zarr_format(fmt: Format) -> str:
    """Encode format as simple string"""
    return str(fmt.__class__.__name__)


def decode_zarr_format(format_name: str) -> Format:
    """Decode format from string"""
    formats = {
        "FormatV01": FormatV01,
        "FormatV02": FormatV02,
        "FormatV03": FormatV03,
        "FormatV04": FormatV04,
        "FormatV05": FormatV05,
    }

    if format_name not in formats:
        raise ValueError(f"Unknown format: {format_name}")

    return formats[format_name]()


class OMEZarrReader:

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
        fmt: Optional[Format] = None,
    ):
        """
        OME-Zarr image reader
        :param input_path: The path to the Zarr image
        """
        self._logger = get_logger_wrapper(False) if not logger else logger
        self._source_ctx = _get_ctx(source_ctx, source_config)
        self._source_cfg = self._source_ctx.config()
        self._dest_ctx = _get_ctx(dest_ctx, dest_config)
        self._dest_cfg = self._dest_ctx.config()

        # Format version
        if not fmt:
            self._fmt = FormatV04()
        else:
            self._fmt = fmt()
        # Format version encoding
        fmt_version = {"format": encode_zarr_format(self._fmt)}
        self._fmt_serial = json.dumps(fmt_version)

        storage_options = translate_config_to_s3fs(self._source_cfg)
        if FormatV05 and isinstance(self._fmt, FormatV05):
            input_fh = FsspecStore.from_url(
                input_path, storage_options=dict(storage_options)
            )
        else:
            input_fh = FsspecStore(input_path, check=True, create=True, **storage_options)
            

        location = ZarrLocation(input_fh, fmt=self._fmt)
        self._root_node = next(Reader(location)())
        self._multiscales = cast(Multiscales, self._root_node.load(Multiscales))
        self._omero = cast(Optional[OMERO], self._root_node.load(OMERO))

    def __enter__(self) -> OMEZarrReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

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
        axes = Axes(a["name"].upper() for a in self._multiscales.node.metadata["axes"])
        self._logger.debug(f"Reader axes: {axes}")
        return axes

    @property
    def channels(self) -> Sequence[str]:
        return (
            tuple(self._omero.node.metadata.get("channel_names", ()))
            if self._omero
            else ()
        )

    @property
    def webp_format(self) -> WebpFilter.WebpInputFormat:
        channels = self._omero.image_data.get("channels", ()) if self._omero else ()
        colors = tuple(channel.get("color") for channel in channels)
        self._logger.debug(f"Webp format - channels: {channels},  colors:{colors}")

        if colors == ("FF0000", "00FF00", "0000FF"):
            return WebpFilter.WebpInputFormat.WEBP_RGB
        return WebpFilter.WebpInputFormat.WEBP_NONE

    @property
    def level_count(self) -> int:
        level_count = len(self._multiscales.datasets)
        self._logger.debug(f"Level count: {level_count}")
        return level_count

    def level_dtype(self, level: int) -> np.dtype:
        dtype = self._multiscales.node.data[level].dtype
        self._logger.debug(f"Level {level} dtype: {dtype}")
        return dtype

    def level_shape(self, level: int) -> Tuple[int, ...]:
        l_shape = cast(Tuple[int, ...], self._multiscales.node.data[level].shape)
        self._logger.debug(f"Level {level} shape: {l_shape}")
        return l_shape

    def level_image(
        self, level: int, tile: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        dask_array = self._multiscales.node.data[level]
        if tile is not None:
            dask_array = dask_array[tile]
        return np.asarray(dask_array)

    def level_metadata(self, level: int) -> Dict[str, Any]:
        dataset = self._multiscales.datasets[level]
        location = ZarrLocation(self._multiscales.zarr.subpath(dataset), fmt=self._fmt)
        zarr_path = os.path.join(
            location.path,
            "zarr.json" if location.fmt.zarr_format == 3 else ".zarray",
        )

        with open(zarr_path, mode="r") as f:
            metadata = json.load(f)

        self._logger.debug(f"Level {level} - Metadata: {json.dumps(metadata)}")
        return {"json_zarray": json.dumps(metadata)}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        multiscale = self._multiscales.lookup("multiscales", [])[0]
        writer_kwargs = dict(
            axes=multiscale.get("axes"),
            coordinate_transformations=[
                d.get("coordinateTransformations") for d in multiscale["datasets"]
            ],
            name=multiscale.get("name"),
            metadata=multiscale.get("metadata"),
            omero=self._omero.image_data if self._omero else None,
            fmt=self._fmt_serial,  # serialized format version
        )
        self._logger.debug(f"Group metadata: {writer_kwargs}")
        return {"json_zarrwriter_kwargs": json.dumps(writer_kwargs)}

    @property
    def image_metadata(self) -> Dict[str, Any]:
        # Based on information available at https://ngff.openmicroscopy.org/latest/#metadata
        # The start and end values may differ from the channel min-max values as well as the
        # min-max values of the metadata.
        metadata: Dict[str, Any] = {}

        base_type = self.level_dtype(0)
        channel_min = (
            np.iinfo(base_type).min
            if np.issubdtype(base_type, numpy.integer)
            else np.finfo(base_type).min
        )
        channel_max = (
            np.iinfo(base_type).max
            if np.issubdtype(base_type, np.integer)
            else np.finfo(base_type).max
        )

        metadata["channels"] = []

        omero_metadata = self._multiscales.lookup("omero", {})
        for idx, channel in enumerate(omero_metadata.get("channels", [])):
            metadata["channels"].append(
                {
                    "id": f"{idx}",
                    "name": channel.get("label", f"Channel:{idx}"),
                    "color": get_rgba(
                        int(
                            channel.get("color", hex(np.random.randint(0, WHITE_RGB)))
                            + "FF",
                            base=16,
                        )
                    ),
                    "min": channel.get("window", {}).get("start", channel_min),
                    "max": channel.get("window", {}).get("end", channel_max),
                }
            )
        self._logger.debug(f"Image metadata: {metadata}")
        return metadata

    @property
    def original_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Dict[str, Any]] = {"ZARR": {}}

        for key, value in self._root_node.root.zarr.root_attrs.items():
            metadata["ZARR"].setdefault(key, value)

        return metadata

    def optimal_reader(
        self, level: int, max_workers: int = 0
    ) -> Optional[Iterator[Tuple[Tuple[slice, ...], NDArray[Any]]]]:
        return None


class OMEZarrWriter:
    def __init__(self, output_path: str, logger: logging.Logger, **kwargs: Any) -> None:
        """
        OME-Zarr image writer from TileDB

        :param output_path: The path to the Zarr image
        """
        self._logger = logger

        metadata = kwargs["metadata"]
        self._group_metadata: Dict[str, Any] = json.loads(
            metadata["json_zarrwriter_kwargs"]
        )

        # Deserialize and decode format version
        deserial_fmt = json.loads(self._group_metadata.pop("fmt"))
        self._fmt = decode_zarr_format(deserial_fmt["format"])

        self._group = zarr.group(
            store=LocalStore(root=output_path),
            overwrite=True,
            zarr_format=self._fmt.zarr_format,
        )
        self._pyramid: List[np.ndarray] = []
        self._storage_options: List[Dict[str, Any]] = []

    def __enter__(self) -> OMEZarrWriter:
        return self

    def write_group_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._group_metadata = self._group_metadata or json.loads(
            metadata["json_zarrwriter_kwargs"]
        )

    def write_level_image(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> None:
        # store the image to be written at __exit__
        self._pyramid.append(image)
        # store the zarray metadata to be written at __exit__
        zarray = dict(metadata)
        # compressor = zarray["compressor"]
        # del compressor["id"]
        # from zarr.codecs import BloscCodec
        # zarray["compressor"] = BloscCodec.from_dict(compressor)
        self._storage_options.append(zarray)

    def compute_level_metadata(
        self,
        baseline: bool,
        num_levels: int,
        image_dtype: np.dtype,
        group_metadata: Mapping[str, Any],
        array_metadata: Mapping[str, Any],
        **writer_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return dict(json.loads(array_metadata.get("json_zarray", "{}")))

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        group_metadata = self._group_metadata

        write_multiscale(
            pyramid=self._pyramid,
            group=self._group,
            fmt=self._fmt,
            axes=group_metadata["axes"],
            coordinate_transformations=group_metadata["coordinate_transformations"],
            storage_options=None,  # https://github.com/ome/ome-zarr-py/pull/480
            name=group_metadata["name"],
            metadata=group_metadata["metadata"] if group_metadata["metadata"] else {},
        )
        if group_metadata["omero"]:
            self._group.attrs["omero"] = group_metadata["omero"]


class OMEZarrConverter(ImageConverterMixin[OMEZarrReader, OMEZarrWriter]):
    """Converter of Zarr-supported images to TileDB Groups of Arrays"""

    _ImageReaderType = OMEZarrReader
    _ImageWriterType = OMEZarrWriter
