from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np

import tiledb
from tiledb.cc import WebpInputFormat

from . import ATTR_NAME
from .converters.axes import Axes
from .converters.scale import Scaler

LENGTH_UNITS = {"m": 0, "dm": -1, "cm": -2, "mm": -3, "μm": -6, "nm": -9, "pm": -12}


class ReadWriteGroup:
    def __init__(self, uri: str):
        parsed_uri = urlparse(uri)
        # normalize uri if it's a local path (e.g. ../..foo/bar)

        # Windows paths produce single letter scheme matching the drive letter
        # Unix absolute path produce an empty scheme
        if len(parsed_uri.scheme) < 2 or parsed_uri.scheme == "file":
            uri = str(Path(parsed_uri.path).resolve()).replace("\\", "/")
        if tiledb.object_type(uri) != "group":
            tiledb.group_create(uri)
        self._uri = uri if uri.endswith("/") else uri + "/"
        self._is_cloud = parsed_uri.scheme == "tiledb"

    def __enter__(self) -> ReadWriteGroup:
        self.r_group = tiledb.Group(self._uri, "r")
        self.w_group = tiledb.Group(self._uri, "w")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.r_group.close()
        self.w_group.close()

    def get_or_create(self, name: str, schema: tiledb.ArraySchema) -> Tuple[str, bool]:
        create = False
        if name in self.r_group:
            uri = self.r_group[name].uri
        else:
            uri = os.path.join(self._uri, name).replace("\\", "/")

            if not tiledb.array_exists(uri):
                tiledb.Array.create(uri, schema)
                create = True
            else:
                # The array exists but it's not added as group member with the given name.
                # It is possible though that it was added as an anonymous member.
                # In this case we should remove the member, using as key either the uri
                # (if added with relative=False) or the name (if added with relative=True).
                for ref in uri, name:
                    try:
                        self.w_group.remove(ref)
                    except tiledb.TileDBError:
                        pass
                    else:
                        # Attempting to remove and then re-add a member with the same name
                        # fails with "[TileDB::Group] Error: Cannot add group member,
                        # member already set for removal.". To work around this we need to
                        # close the write group (to flush the removal) and and reopen it
                        # (to allow the add operation)
                        self.w_group.close()
                        self.w_group.open("w")
            # register the uri with the given name
            if self._is_cloud:
                self.w_group.add(uri, name, relative=False)
            else:
                self.w_group.add(name, name, relative=True)
        return uri, create


def open_bioimg(uri: str, mode: str = "r", attr: str = ATTR_NAME) -> tiledb.Array:
    return tiledb.open(uri, mode=mode, attr=attr if mode == "r" else None)


def get_schema(
    dim_names: Tuple[str, ...],
    dim_shape: Tuple[int, ...],
    max_tiles: Mapping[str, int],
    attr_dtype: np.dtype,
    compressor: tiledb.Filter,
) -> tiledb.ArraySchema:
    # find the smallest dtype that can hold `np.prod(dim_shape)` values
    dim_dtype = np.min_scalar_type(np.prod(dim_shape))

    dims = []
    assert len(dim_names) == len(dim_shape), (dim_names, dim_shape)
    for dim_name, dim_size in zip(dim_names, dim_shape):
        dim_tile = min(dim_size, max_tiles[dim_name])
        dim = tiledb.Dim(dim_name, (0, dim_size - 1), dim_tile, dtype=dim_dtype)
        dims.append(dim)
    attr = tiledb.Attr(name=ATTR_NAME, dtype=attr_dtype, filters=[compressor])
    return tiledb.ArraySchema(domain=tiledb.Domain(*dims), attrs=[attr])


def create_image_pyramid(
    rw_group: ReadWriteGroup,
    base_uri: str,
    base_level: int,
    max_tiles: Mapping[str, int],
    compressor: tiledb.Filter,
    pyramid_kwargs: Mapping[str, Any],
) -> None:
    with open_bioimg(base_uri) as a:
        base_shape = a.shape
        dim_names = tuple(dim.name for dim in a.domain)
        dim_axes = "".join(dim_names)
        attr_dtype = a.attr(0).dtype

    scaler = Scaler(base_shape, dim_axes, **pyramid_kwargs)
    for i, dim_shape in enumerate(scaler.level_shapes):
        level = base_level + 1 + i
        schema = get_schema(dim_names, dim_shape, max_tiles, attr_dtype, compressor)
        uri, created = rw_group.get_or_create(f"l_{level}.tdb", schema)
        if not created:
            continue

        with open_bioimg(uri, mode="w") as out_array:
            out_array.meta.update(level=level)
            with open_bioimg(base_uri) as in_array:
                scaler.apply(in_array, out_array, i)

        # if a non-progressive method is used, the input layer of the scaler
        # is the base image layer else we use the previously generated layer
        if scaler.progressive:
            base_uri = uri


def iter_levels_meta(group: tiledb.Group) -> Iterator[Mapping[str, Any]]:
    for o in group:
        with open_bioimg(o.uri) as array:
            level = array.meta["level"]
            domain = array.schema.domain
            axes = Axes(
                "".join(domain.dim(dim_idx).name for dim_idx in range(domain.ndim))
            )
            shape = (
                axes.webp_mapper(4).inverse.map_shape(array.shape)
                if (
                    isinstance(
                        array.schema.attr(ATTR_NAME).filters[0], tiledb.WebpFilter
                    )
                )
                else array.shape
            )
            yield dict(level=level, name=f"l_{level}.tdb", axes=axes.dims, shape=shape)


def get_pixel_depth(compressor: tiledb.Filter) -> int:
    if not isinstance(compressor, tiledb.WebpFilter):
        return 1
    webp_format = compressor.input_format
    if webp_format in (WebpInputFormat.WEBP_RGB, WebpInputFormat.WEBP_BGR):
        return 3
    if webp_format in (WebpInputFormat.WEBP_RGBA, WebpInputFormat.WEBP_BGRA):
        return 4
    raise ValueError(f"Invalid WebpInputFormat: {compressor.input_format}")


def get_axes_mapping(
    compressor: tiledb.Filter, axes: str
) -> Mapping[str, Sequence[str]]:
    if isinstance(compressor, tiledb.WebpFilter):
        return {"Y": ["Y"], "X": ["X", "C"]}

    return {axis: [axis] for axis in axes}


def iter_color(attr_type: np.dtype) -> Iterator[Mapping[str, Any]]:
    yield {
        "red": np.iinfo(attr_type).max,
        "green": np.iinfo(attr_type).min,
        "blue": np.iinfo(attr_type).min,
        "alpha": np.iinfo(attr_type).max,
    }
    yield {
        "red": np.iinfo(attr_type).min,
        "green": np.iinfo(attr_type).max,
        "blue": np.iinfo(attr_type).min,
        "alpha": np.iinfo(attr_type).max,
    }
    yield {
        "red": np.iinfo(attr_type).min,
        "green": np.iinfo(attr_type).min,
        "blue": np.iinfo(attr_type).max,
        "alpha": np.iinfo(attr_type).max,
    }

    while True:
        if np.issubdtype(attr_type, np.integer):
            red = np.random.randint(
                low=np.iinfo(attr_type).min,
                high=np.iinfo(attr_type).max,
                dtype=attr_type,
            )
            green = np.random.randint(
                low=np.iinfo(attr_type).min,
                high=np.iinfo(attr_type).max,
                dtype=attr_type,
            )
            blue = np.random.randint(
                low=np.iinfo(attr_type).min,
                high=np.iinfo(attr_type).max,
                dtype=attr_type,
            )
        else:
            red = np.random.uniform(
                low=np.iinfo(attr_type).min, high=np.iinfo(attr_type).max
            ).astype(attr_type)
            green = np.random.uniform(
                low=np.iinfo(attr_type).min, high=np.iinfo(attr_type).max
            ).astype(attr_type)
            blue = np.random.uniform(
                low=np.iinfo(attr_type).min, high=np.iinfo(attr_type).max
            ).astype(attr_type)

        yield {
            "red": red,
            "green": green,
            "blue": blue,
            "alpha": np.iinfo(attr_type).max,
        }


def get_rgba(value: int) -> dict[str, int]:
    color = {
        "red": (value & 0xFF000000) // 2**24,
        "green": (value & 0x00FF0000) // 2**16,
        "blue": (value & 0x0000FF00) // 2**8,
        "alpha": value & 0x000000FF,
    }

    return color


def length_converter(value: float, original_unit: str, requested_unit: str) -> float:
    return value * 10 ** (LENGTH_UNITS[original_unit] - LENGTH_UNITS[requested_unit])
