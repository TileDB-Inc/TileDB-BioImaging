import io
import platform
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Mapping, Sequence, Tuple

import cv2
import flask
import numpy as np
from flask import Flask, Response, jsonify, make_response
from flask_cors import CORS
from PIL import Image
from werkzeug.serving import run_simple

import tiledb
from tiledb.bioimg.converters.axes import Axes, AxesMapper


@dataclass
class ZoomLevelRecord:
    zoom_level: int
    image_level: int
    downsample: int
    width: int
    height: int
    array_uri: str


def calculate_slice(
    z: int,
    y: int,
    x: int,
    tile_width: int,
    tile_height: int,
    downsample: int,
    arrays: Mapping[int, tiledb.Array],
) -> Tuple[Tuple[slice, ...], int, int]:
    a = arrays[z]
    dims = "".join(dim.name for dim in a.domain)

    dimx = a.schema.domain.dim("X")
    dimy = a.schema.domain.dim("Y")
    tx = tile_width * downsample
    ty = tile_height * downsample

    maxx = dimx.domain[1]
    maxy = dimy.domain[1]

    sx0 = int(tx * x)
    sx1 = int(sx0 + tx)
    sx1 = np.min((sx1, maxx)).astype(int)

    sy0 = int(ty * y)
    sy1 = int(sy0 + ty)
    sy1 = np.min((sy1, maxy)).astype(int)
    if (sx0 < 0 or sy0 < 0) or (sx0 > maxx or sy0 > maxy):
        flask.abort(404)

    dim_to_slice = {"X": slice(sx0, sx1), "Y": slice(sy0, sy1)}
    ranges = tuple(dim_to_slice.get(dim, slice(None)) for dim in dims)

    return ranges, sx1 - sx0, sy1 - sy0


def calculate_metadata(image_uri: str) -> Sequence[ZoomLevelRecord]:
    metadata = []
    dimensions = []
    uris = []

    group = tiledb.Group(image_uri, "r")
    for member in group:
        with tiledb.open(member.uri) as array:
            axes_mapper = AxesMapper(
                Axes("".join(dim.name for dim in array.domain)), Axes("XYC")
            )
            dimensions.append(axes_mapper.map_shape(array.shape))
            uris.append(member.uri)

    dimensions, uris = (list(t) for t in zip(*sorted(zip(dimensions, uris))))

    current_dim = dimensions.pop(0)
    metadata.append(
        ZoomLevelRecord(
            zoom_level=0,
            image_level=0,
            downsample=1,
            width=current_dim[0],
            height=current_dim[1],
            array_uri=uris.pop(0),
        )
    )
    current_zoom = 2
    image_level = 1
    zoom_level = 0
    while dimensions:
        zoom_level += 1
        metadata.append(
            ZoomLevelRecord(
                zoom_level=zoom_level,
                image_level=image_level,
                downsample=1,
                width=dimensions[0][0],
                height=dimensions[0][1],
                array_uri=uris[0],
            )
        )
        current_zoom_diff = current_zoom * current_dim[0] - dimensions[0][0]
        if abs(current_zoom_diff) < current_zoom:
            current_zoom = 2
            image_level += 1
            current_dim = dimensions.pop(0)
            del uris[0]
        else:
            if current_zoom_diff > 0:
                print("Pyramid level dimension are not multiples of 2")
                break
            current_zoom *= 2

    for i in range(len(metadata) - 2, -1, -1):
        if metadata[i].image_level == metadata[i + 1].image_level:
            metadata[i].downsample = 2 * metadata[i + 1].downsample

    return metadata


class TileServer(object):
    def info(self) -> Response:
        return jsonify(
            {
                "minZoom": min(r.zoom_level for r in self._metadata),
                "maxZoom": max(r.zoom_level for r in self._metadata),
                "dimensions": [(r.width, r.height) for r in self._metadata],
            }
        )

    def tile(
        self,
        z: int,
        y: int,
        x: int,
        tile_width: int,
        tile_height: int,
    ) -> Response:
        zlr = self._metadata[z]
        ranges, sx, sy = calculate_slice(
            zlr.image_level,
            y,
            x,
            tile_width,
            tile_height,
            zlr.downsample,
            self._arrays,
        )
        a = self._arrays[zlr.image_level]

        axes_mapper = AxesMapper(
            Axes("".join(dim.name for dim in a.domain)), Axes("YXC")
        )
        data = axes_mapper.map_array(a[ranges])
        if zlr.downsample != 1:
            data = cv2.resize(
                data,
                (sx // zlr.downsample, sy // zlr.downsample),
                interpolation=cv2.INTER_NEAREST,
            )

        im = Image.fromarray(data).convert("RGB")
        if im.size != (tile_width, tile_height):
            # pad to tile size
            im2 = Image.new("RGB", (tile_width, tile_height), color="white")
            im2.paste(im)
            im = im2

        bdata = io.BytesIO()
        im.save(bdata, "JPEG", quality=70, subsampling=0)
        response = make_response(bdata.getvalue())
        response.headers["Content-Type"] = "image/jpeg"
        response.headers["Content-Disposition"] = "filename=%d.jpeg" % tile_width

        return response

    def __init__(self, name: str, image_uri: str):
        self._app = Flask(name)
        self._pool = Pool(1)
        self._image_uri = image_uri
        self._metadata = calculate_metadata(image_uri)
        self._arrays = {
            record.image_level: tiledb.open(record.array_uri)
            for record in self._metadata
        }
        CORS(self._app)
        self._app.add_url_rule(
            "/api/<int:z>/<int:y>/<int:x>/<int:tile_width>/<int:tile_height>.jpeg",
            "tile",
            self.tile,
        )
        self._app.add_url_rule("/api/info", "info", self.info)

    def __del__(self) -> None:
        self._pool.terminate()

    def run(self, host: str = "0.0.0.0", port: int = 8000, processes: int = 1) -> None:
        if platform.system() == "Windows":
            self._pool.apply_async(self._app.run, kwds={"host": host, "port": port})
        else:
            kwds = {
                "hostname": host,
                "port": port,
                "application": self._app,
            }
            if processes > 1:
                kwds["processes"] = processes
            else:
                kwds["threaded"] = True
            self._pool.apply_async(run_simple, kwds=kwds)

    def terminate(self) -> None:
        self._pool.terminate()
