from __future__ import annotations

from typing import Any, Type

from .converters.base import ImageConverter
from .converters.ome_tiff import OMETiffConverter
from .converters.ome_zarr import OMEZarrConverter
from .converters.openslide import OpenSlideConverter
from .helpers import get_logger_wrapper
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
) -> Type[ImageConverter]:
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
    if converter is Converters.OMETIFF:
        logger.info("Converting OME-TIFF file")
        return OMETiffConverter.to_tiledb(
            source=src,
            output_path=dest,
            log=logger,
            exclude_metadata=exclude_metadata,
            tile_scale=tile_scale,
            **kwargs,
        )
    elif converter is Converters.OMEZARR:
        logger.info("Converting OME-Zarr file")
        return OMEZarrConverter.to_tiledb(
            source=src,
            output_path=dest,
            log=logger,
            exclude_metadata=exclude_metadata,
            tile_scale=tile_scale,
            **kwargs,
        )
    else:
        logger.info("Converting Openslide")
        return OpenSlideConverter.to_tiledb(
            source=src,
            output_path=dest,
            log=logger,
            exclude_metadata=exclude_metadata,
            tile_scale=tile_scale,
            **kwargs,
        )


def to_bioimg(
    src: str,
    dest: str,
    converter: Converters = Converters.OMETIFF,
    *,
    verbose: bool = False,
    **kwargs: Any,
) -> Type[ImageConverter]:
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
    logger = get_logger_wrapper(verbose)
    if converter is Converters.OMETIFF:
        logger.info("Converting to OME-TIFF file")
        return OMETiffConverter.from_tiledb(
            input_path=src, output_path=dest, log=logger, **kwargs
        )
    elif converter is Converters.OMEZARR:
        logger.info("Converting to OME-Zarr file")
        return OMEZarrConverter.from_tiledb(
            input_path=src, output_path=dest, log=logger, **kwargs
        )
    else:
        raise NotImplementedError(
            "Openslide Converter does not support exportation back to bio-imaging formats"
        )
