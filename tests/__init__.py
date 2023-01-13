import multiprocessing
import os
from pathlib import Path

import numpy as np

import tiledb

if os.name == "posix":
    multiprocessing.set_start_method("forkserver")


DATA_DIR = Path(__file__).parent / "data"


def get_schema(x_size, y_size, c_size=3):
    return tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim("C", (0, c_size - 1), tile=c_size, dtype=np.uint32),
            tiledb.Dim("Y", (0, y_size - 1), tile=min(y_size, 1024), dtype=np.uint32),
            tiledb.Dim("X", (0, x_size - 1), tile=min(x_size, 1024), dtype=np.uint32),
        ),
        attrs=[
            tiledb.Attr(
                dtype=np.uint8,
                filters=tiledb.FilterList([tiledb.ZstdFilter(level=0)]),
            )
        ],
    )


def get_path(uri):
    return DATA_DIR / uri
