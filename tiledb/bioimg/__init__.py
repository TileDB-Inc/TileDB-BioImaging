ATTR_NAME = "intensity"
WHITE_RGB = 16777215  # FFFFFF in hex
WHITE_RGBA = 4294967295  # FFFFFFFF in hex
EXPORT_TILE_SIZE = 256
WIN_OPENSLIDE_PATH = r"C:\openslide-win64\bin"

import importlib.util
import os
import warnings
from typing import Optional
from .types import Converters

_osd_exc: Optional[ImportError]

if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(WIN_OPENSLIDE_PATH):
        try:
            importlib.util.find_spec("openslide")
        except ImportError as err_osd:
            warnings.warn(
                "Openslide Converter requires 'openslide-python' package. "
                "You can install 'tiledb-bioimg' with the 'openslide' or 'full' flag"
            )
            _osd_exc = err_osd
        else:
            _osd_exc = None
else:
    try:
        importlib.util.find_spec("openslide")
    except ImportError as err_osd:
        warnings.warn(
            "Openslide Converter requires 'openslide-python' package. "
            "You can install 'tiledb-bioimg' with the 'openslide' or 'full' flag"
        )
        _osd_exc = err_osd
    else:
        _osd_exc = None

from .wrappers import *
