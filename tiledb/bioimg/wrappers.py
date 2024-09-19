from __future__ import annotations

import importlib.util
from typing import Any, Optional

try:
    importlib.util.find_spec("tifffile")
    importlib.util.find_spec("imagecodecs")
except ImportError as err_tiff:
    _tiff_exc: Optional[ImportError] = err_tiff
else:
    _tiff_exc = None
try:
    importlib.util.find_spec("zarr")
    importlib.util.find_spec("ome-zarr")
except ImportError as err_zarr:
    _zarr_exc: Optional[ImportError] = err_zarr
else:
    _zarr_exc = None

from . import _osd_exc
from .helpers import get_logger_wrapper
from .plugin_manager import load_converters
from .types import Converters


def from_bioimg(
    src: str,
    dest: str,
    converter: Converters = Converters.OMETIFF,
    *,
    verbose: bool = False,
    exclude_metadata: bool = False,
    tile_scale: int = 1,
    **kwargs: Any,
) -> Any:
    """
    This function is a wrapper and serves as an all-inclusive API for encapsulating the
    ingestion of different file formats
    :param src: The source path for the file to be ingested *.tiff, *.zarr, *.svs etc..
    :param dest: The destination path where the TileDB image will be stored
    :param converter: The converter type to be used (tentative) soon automatically detected
    :param verbose: verbose logging, defaults to False
    :param kwargs: keyword arguments for custom ingestion behaviour
    :return: The converter class that was used for the ingestion
    """

    logger = get_logger_wrapper(verbose)
    reader_kwargs = kwargs.get("reader_kwargs", {})

    # Get the config for the source
    reader_kwargs["source_config"] = kwargs.pop("source_config", None)

    # Get the config for the destination (if exists) otherwise match it with source config
    reader_kwargs["dest_config"] = kwargs.pop(
        "dest_config", reader_kwargs["source_config"]
    )
    converters = load_converters()
    if converter is Converters.OMETIFF:
        if not _tiff_exc:
            logger.info("Converting OME-TIFF file")
            return converters["tiff_converter"].to_tiledb(
                source=src,
                output_path=dest,
                log=logger,
                exclude_metadata=exclude_metadata,
                tile_scale=tile_scale,
                reader_kwargs=reader_kwargs,
                **kwargs,
            )
        else:
            raise _tiff_exc
    elif converter is Converters.OMEZARR:
        if not _zarr_exc:
            logger.info("Converting OME-Zarr file")
            return converters["zarr_converter"].to_tiledb(
                source=src,
                output_path=dest,
                log=logger,
                exclude_metadata=exclude_metadata,
                tile_scale=tile_scale,
                reader_kwargs=reader_kwargs,
                **kwargs,
            )
        else:
            raise _zarr_exc
    elif converter is Converters.OSD:
        if not _osd_exc:
            logger.info("Converting Openslide")
            return converters["osd_converter"].to_tiledb(
                source=src,
                output_path=dest,
                log=logger,
                exclude_metadata=exclude_metadata,
                tile_scale=tile_scale,
                reader_kwargs=reader_kwargs,
                **kwargs,
            )
        else:
            raise _osd_exc
    else:

        logger.info("Converting PNG")
        return converters["png_converter"].to_tiledb(
            source=src,
            output_path=dest,
            log=logger,
            exclude_metadata=exclude_metadata,
            tile_scale=tile_scale,
            reader_kwargs=reader_kwargs,
            **kwargs,
        )


def to_bioimg(
    src: str,
    dest: str,
    converter: Converters = Converters.OMETIFF,
    *,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """
    This function is a wrapper and serves as an all-inclusive API for encapsulating the
    exportation of TileDB ingested bio-images back into different file formats
    :param src: The source path where the TileDB image is stored
    :param dest: The destination path for the image file to be exported *.tiff, *.zarr, *.svs etc..
    :param converter: The converter type to be used
    :param verbose: verbose logging, defaults to False
    :param kwargs: keyword arguments for custom exportation behaviour
    :return: None
    """
    converters = load_converters()
    logger = get_logger_wrapper(verbose)
    if converter is Converters.OMETIFF:
        if not _tiff_exc:
            logger.info("Converting to OME-TIFF file")
            return converters["tiff_converter"].from_tiledb(
                input_path=src, output_path=dest, log=logger, **kwargs
            )
        else:
            raise _tiff_exc
    elif converter is Converters.OMEZARR:
        if not _zarr_exc:
            logger.info("Converting to OME-Zarr file")
            return converters["zarr_converter"].from_tiledb(
                input_path=src, output_path=dest, log=logger, **kwargs
            )
        else:
            raise _zarr_exc
    elif converter is Converters.PNG:
        logger.info("Converting to PNG file")
        return converters["png_converter"].from_tiledb(
            input_path=src, output_path=dest, log=logger, **kwargs
        )
    else:
        raise NotImplementedError(
            "Openslide Converter does not support exportation back to bio-imaging formats"
        )
