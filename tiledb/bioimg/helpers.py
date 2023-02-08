from __future__ import annotations

import os
from typing import Any, Iterator, Mapping, Optional, Tuple

import numpy as np

import tiledb
from tiledb.cc import WebpInputFormat

from . import ATTR_NAME
from .converters.scale import Scaler


class _ReadWriteGroup:
    def __init__(self, uri: str):
        if tiledb.object_type(uri) != "group":
            tiledb.group_create(uri)
        self.uri = uri

    def __enter__(self) -> _ReadWriteGroup:
        self.r_group = tiledb.Group(self.uri, "r")
        self.w_group = tiledb.Group(self.uri, "w")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.r_group.close()
        self.w_group.close()

    def get_or_create(self, name: str, schema: tiledb.ArraySchema) -> Tuple[str, bool]:
        create = False
        if name in self.r_group:
            uri = self.r_group[name].uri
        else:
            uri = os.path.join(self.r_group.uri, name)
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
            self.w_group.add(uri, name)
        return uri, create


def _open(uri: str, *args: Optional[Any], **kwargs: Optional[Any]) -> tiledb.Array:
    attr = None if "w" in args or kwargs.get("mode") == "w" else ATTR_NAME
    return tiledb.open(uri, attr=attr, *args, **kwargs)


def _get_schema(
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
    attr = tiledb.Attr(name="intensity", dtype=attr_dtype, filters=[compressor])
    return tiledb.ArraySchema(domain=tiledb.Domain(*dims), attrs=[attr])


def _create_image_pyramid(
    rw_group: _ReadWriteGroup,
    base_uri: str,
    base_level: int,
    max_tiles: Mapping[str, int],
    compressor: tiledb.Filter,
    pyramid_kwargs: Mapping[str, Any],
) -> None:
    with _open(base_uri) as a:
        base_shape = a.shape
        dim_names = tuple(dim.name for dim in a.domain)
        dim_axes = "".join(dim_names)
        attr_dtype = a.attr(0).dtype

    scaler = Scaler(base_shape, dim_axes, **pyramid_kwargs)
    for i, dim_shape in enumerate(scaler.level_shapes):
        level = base_level + 1 + i
        schema = _get_schema(dim_names, dim_shape, max_tiles, attr_dtype, compressor)
        uri, created = rw_group.get_or_create(f"l_{level}.tdb", schema)
        if not created:
            continue

        with _open(uri, mode="w") as out_array:
            out_array.meta.update(level=level)
            with _open(base_uri) as in_array:
                scaler.apply(in_array, out_array, i)

        # if a non-progressive method is used, the input layer of the scaler
        # is the base image layer else we use the previously generated layer
        if scaler.progressive:
            base_uri = uri


def _iter_levels_meta(group: tiledb.Group) -> Iterator[Mapping[str, Any]]:
    for o in group:
        with _open(o.uri) as array:
            level = array.meta["level"]
            domain = array.schema.domain
            axes = "".join(domain.dim(dim_idx).name for dim_idx in range(domain.ndim))
            yield dict(level=level, name=f"l_{level}.tdb", axes=axes, shape=array.shape)


def _get_pixel_depth(compressor: tiledb.Filter) -> int:
    if not isinstance(compressor, tiledb.WebpFilter):
        return 1
    webp_format = compressor.input_format
    if webp_format in (WebpInputFormat.WEBP_RGB, WebpInputFormat.WEBP_BGR):
        return 3
    if webp_format in (WebpInputFormat.WEBP_RGBA, WebpInputFormat.WEBP_BGRA):
        return 4
    raise ValueError(f"Invalid WebpInputFormat: {compressor.input_format}")
