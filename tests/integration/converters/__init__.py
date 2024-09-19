import os
from tiledb.bioimg import WIN_OPENSLIDE_PATH

if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    os.add_dll_directory(WIN_OPENSLIDE_PATH)
