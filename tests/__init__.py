import multiprocessing
import os
import urllib.parse

import boto3
import numpy as np
import tiledb

if os.name == "posix":
    multiprocessing.set_start_method("forkserver")


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


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


def get_path(uri: str) -> str:
    if uri.startswith("s3://"):
        s3 = boto3.client("s3")
        parsed_uri = urllib.parse.urlparse(uri)
        local_path = os.path.join(DATA_DIR, os.path.basename(parsed_uri.path))
        if not os.path.exists(local_path):
            s3.download_file(parsed_uri.netloc, parsed_uri.path.lstrip("/"), local_path)
    else:
        local_path = os.path.join(DATA_DIR, uri)
    return local_path
