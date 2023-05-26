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
    if converter is Converters.OMETIFF:
        OMETiffConverter.to_tiledb(source=src, output_path=dest, **kwargs)
        return OMETiffConverter
    elif converter is Converters.OMEZARR:
        OMEZarrConverter.to_tiledb(source=src, output_path=dest, **kwargs)
        return OMEZarrConverter
    else:
        OpenSlideConverter.to_tiledb(source=src, output_path=dest, **kwargs)
        return OpenSlideConverter
