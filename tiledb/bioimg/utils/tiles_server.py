import io
import platform

# import glob
from dataclasses import dataclass

# from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import flask
import numpy as np
from flask import Flask, Response, jsonify, make_response
from flask_cors import CORS
from multiprocess import Pool
from PIL import Image
from werkzeug.serving import run_simple

import tiledb

from ..converters.axes import Axes, AxesMapper


@dataclass
class ZoomLevelRecord:
    zoom_level: int
    image_level: int
    downsample: int
    width: int
    height: int
    array_uri: str


class TileHelpers:
    @staticmethod
    def calculate_slice(
        z: int,
        y: int,
        x: int,
        tile_width: int,
        tile_height: int,
        downsample: int,
        arrays: Dict[int, tiledb.array.DenseArray],
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

    @staticmethod
    def calculate_metadata(image_uri: str) -> List[ZoomLevelRecord]:
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

        metadata.append(
            ZoomLevelRecord(
                zoom_level=0,
                image_level=0,
                downsample=1,
                width=dimensions[0][0],
                height=dimensions[0][1],
                array_uri=uris[0],
            )
        )
        current_dim = dimensions[0]
        dimensions.pop(0)
        uris.pop(0)
        current_zoom = 2
        current_level = 1
        level = 0
        while len(dimensions) > 0:
            level += 1
            metadata.append(
                ZoomLevelRecord(
                    zoom_level=level,
                    image_level=current_level,
                    downsample=1,
                    width=dimensions[0][0],
                    height=dimensions[0][1],
                    array_uri=uris[0],
                )
            )

            if abs(current_zoom * current_dim[0] - dimensions[0][0]) < current_zoom:
                current_zoom = 2
                current_level += 1
                current_dim = dimensions[0]
                dimensions.pop(0)
                uris.pop(0)
            else:
                if current_zoom * current_dim[0] > dimensions[0][0]:
                    print("Pyramid level dimension are not multiples of 2")
                    break

                current_zoom *= 2

        for i in range(len(metadata) - 2, -1, -1):
            if metadata[i].image_level == metadata[i + 1].image_level:
                metadata[i].downsample = 2 * metadata[i + 1].downsample

        return metadata


class API:
    @staticmethod
    def info(metadata: List[ZoomLevelRecord]) -> Response:
        info = {
            "minZoom": min(metadata, key=lambda r: r.zoom_level).zoom_level,
            "maxZoom": max(metadata, key=lambda r: r.zoom_level).zoom_level,
            "dimensions": [(record.width, record.height) for record in metadata],
        }

        return jsonify(info)

    @staticmethod
    def tile(
        z: int,
        y: int,
        x: int,
        tile_width: int,
        tile_height: int,
        metadata: List[ZoomLevelRecord],
        arrays: Dict[int, tiledb.array.DenseArray],
    ) -> Response:
        ranges, sx, sy = TileHelpers.calculate_slice(
            metadata[z].image_level,
            y,
            x,
            tile_width,
            tile_height,
            metadata[z].downsample,
            arrays,
        )
        a = arrays[metadata[z].image_level]

        axes_mapper = AxesMapper(
            Axes("".join(dim.name for dim in a.domain)), Axes("YXC")
        )
        data = axes_mapper.map_array(a[ranges])
        if metadata[z].downsample != 1:
            data = cv2.resize(
                data,
                (sx // metadata[z].downsample, sy // metadata[z].downsample),
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


class EndpointAction(object):
    def __init__(self, action: Callable[..., Response], **kwargs: Any):
        self.action = action
        self.additional_args = kwargs

    def __call__(self, **kwargs: Any) -> Response:
        return self.action(**({**kwargs, **self.additional_args}))


class TileServer(object):
    def __init__(self, name: str, image_uri: str):
        self._app = Flask(name)
        self._pool = Pool(1)
        self._image_uri = image_uri
        self._metadata = TileHelpers.calculate_metadata(image_uri)
        self._arrays = {
            record.image_level: tiledb.open(record.array_uri)
            for record in self._metadata
        }

        CORS(self._app)

    def __del__(self) -> None:
        self._pool.terminate()

    def run(
        self, host: str = "0.0.0.0", port: int = 8000, multiprocess: bool = False
    ) -> None:
        print(platform.system())
        if platform.system() == "Windows":
            self._pool.apply_async(self._app.run, kwds={"host": host, "port": port})
        else:
            if multiprocess:
                self._pool.apply_async(
                    run_simple,
                    kwds={
                        "hostname": host,
                        "port": port,
                        "application": self._app,
                        "threaded": False,
                        "processes": 8,
                    },
                )
            else:
                self._pool.apply_async(
                    run_simple,
                    kwds={
                        "hostname": host,
                        "port": port,
                        "application": self._app,
                        "threaded": True,
                        "processes": 1,
                    },
                )

    def terminate(self) -> None:
        self._pool.terminate()

    def initialize(self) -> None:
        self.add_endpoint(
            endpoint="/api/<int:z>/<int:y>/<int:x>/<int:tile_width>/<int:tile_height>.jpeg",
            endpoint_name="tile",
            handler=API.tile,
            metadata=self._metadata,
            arrays=self._arrays,
        )

        self.add_endpoint(
            "/api/info", endpoint_name="info", handler=API.info, metadata=self._metadata
        )

    def add_endpoint(
        self,
        endpoint: str,
        endpoint_name: str,
        handler: Callable[..., Response],
        **kwargs: Any,
    ) -> None:
        self._app.add_url_rule(
            endpoint, endpoint_name, EndpointAction(handler, **kwargs)
        )
