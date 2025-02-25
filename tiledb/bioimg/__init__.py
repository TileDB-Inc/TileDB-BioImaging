ATTR_NAME = "intensity"
WHITE_RGB = 16777215  # FFFFFF in hex
WHITE_RGBA = 4294967295  # FFFFFFFF in hex
EXPORT_TILE_SIZE = 256

import importlib.util
import warnings
from typing import Optional
from .types import Converters

_osd_exc: Optional[ImportError]
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
