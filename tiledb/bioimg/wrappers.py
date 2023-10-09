from __future__ import annotations

from typing import Any, Type

from .converters.base import ImageConverter
from .converters.ome_tiff import OMETiffConverter
from .converters.ome_zarr import OMEZarrConverter
from .converters.openslide import OpenSlideConverter
from .types import Converters


def from_bioimg(
    src: str, dest: str, converter: Converters = Converters.OMETIFF, **kwargs: Any
) -> Type[ImageConverter]:
    """This function is a wrapper and serves as an all-inclusive API for encapsulating the
    ingestion of different file formats

    Parameters
    ----------
    src :
        The source path for the file to be ingested *.tiff, *.zarr, *.svs etc..
    dest :
        The destination path where the TileDB image will be stored
    converter :
        The converter type to be used (tentative) soon automatically detected
    kwargs :
        keyword arguments for custom ingestion behaviour
    src: str :
        
    dest: str :
        
    converter: Converters :
         (Default value = Converters.OMETIFF)
    **kwargs: Any :
        

    Returns
    -------
    type
        The converter class that was used for the ingestion

    """
    if converter is Converters.OMETIFF:
        return OMETiffConverter.to_tiledb(source=src, output_path=dest, **kwargs)
    elif converter is Converters.OMEZARR:
        return OMEZarrConverter.to_tiledb(source=src, output_path=dest, **kwargs)
    else:
        return OpenSlideConverter.to_tiledb(source=src, output_path=dest, **kwargs)


def to_bioimg(
    src: str, dest: str, converter: Converters = Converters.OMETIFF, **kwargs: Any
) -> Type[ImageConverter]:
    """This function is a wrapper and serves as an all-inclusive API for encapsulating the
    exportation of TileDB ingested bio-images back into different file formats

    Parameters
    ----------
    src :
        The source path where the TileDB image is stored
    dest :
        The destination path for the image file to be exported *.tiff, *.zarr, *.svs etc..
    converter :
        The converter type to be used
    kwargs :
        keyword arguments for custom exportation behaviour
    src: str :
        
    dest: str :
        
    converter: Converters :
         (Default value = Converters.OMETIFF)
    **kwargs: Any :
        

    Returns
    -------
    type
        None

    """

    if converter is Converters.OMETIFF:
        return OMETiffConverter.from_tiledb(input_path=src, output_path=dest, **kwargs)
    elif converter is Converters.OMEZARR:
        return OMEZarrConverter.from_tiledb(input_path=src, output_path=dest, **kwargs)
    else:
        raise NotImplementedError(
            "Openslide Converter does not support exportation back to bio-imaging formats"
        )
