import multiprocessing
import os
import urllib.parse
from pathlib import Path

import boto3
import numpy as np

import tiledb

if os.name == "posix":
    multiprocessing.set_start_method("forkserver")


DATA_DIR = Path(__file__).parent / "data"


def get_schema(x_size, y_size):
    return tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim("C", (0, 2), tile=3, dtype=np.uint32),
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
    if uri.startswith("s3://"):
        s3 = boto3.client("s3")
        parsed_uri = urllib.parse.urlparse(uri)
        local_path = DATA_DIR / Path(parsed_uri.path).name
        if not local_path.exists():
            s3.download_file(parsed_uri.netloc, parsed_uri.path.lstrip("/"), local_path)
    else:
        local_path = DATA_DIR / uri
    return local_path
