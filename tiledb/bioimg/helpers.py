from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np

import tiledb
from tiledb import Config
from tiledb.cc import WebpInputFormat

from . import ATTR_NAME
from .converters.axes import Axes, AxesMapper
from .version import version_tuple


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
                # The array exists, but it's not added as group member with the given name.
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


def open_bioimg(
    uri: str, mode: str = "r", attr: str = ATTR_NAME, config: Config = None
) -> tiledb.Array:
    return tiledb.open(
        uri, mode=mode, attr=attr if mode == "r" else None, config=config
    )


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
    # WEBP Compressor does not accept specific dtypes so for dimensions we use the default
    dim_compressor = tiledb.ZstdFilter(level=0)
    if not isinstance(compressor, tiledb.WebpFilter):
        dim_compressor = compressor
    for dim_name, dim_size in zip(dim_names, dim_shape):
        dim_tile = min(dim_size, max_tiles[dim_name])
        dim = tiledb.Dim(
            dim_name,
            (0, dim_size - 1),
            dim_tile,
            dtype=dim_dtype,
            filters=[dim_compressor],
        )
        dims.append(dim)
    attr = tiledb.Attr(name=ATTR_NAME, dtype=attr_dtype, filters=[compressor])
    return tiledb.ArraySchema(domain=tiledb.Domain(*dims), attrs=[attr])


def get_axes_mapper(
    source_axes: Axes,
    source_shape: Tuple[int, ...],
    compressor: Mapping[int, tiledb.Filter],
    level: int,
    preserve_axes: bool,
    max_tiles: Mapping[str, int],
) -> Tuple[AxesMapper, Tuple[str, ...], MutableMapping[str, int]]:
    tiles = dict(max_tiles)
    pixel_depth = get_pixel_depth(compressor.get(level, tiledb.ZstdFilter(level=0)))
    if pixel_depth == 1:
        if preserve_axes:
            target_axes = source_axes
        else:
            target_axes = source_axes.canonical(source_shape)
        axes_mapper = source_axes.mapper(target_axes)
        dim_names = tuple(target_axes.dims)
    else:
        tiles["X"] *= pixel_depth
        axes_mapper = source_axes.webp_mapper(pixel_depth)
        dim_names = ("Y", "X")

    return axes_mapper, dim_names, tiles


def iter_levels_meta(group: tiledb.Group) -> Iterator[Mapping[str, Any]]:
    for o in group:
        with open_bioimg(o.uri) as array:
            try:
                level = array.meta["level"]
            except KeyError as exc:
                raise RuntimeError(
                    "Key: 'level' not found in array metadata. Make sure that levels have been "
                    "ingested correctly in any previous process for the same image."
                ) from exc
            domain = array.schema.domain
            axes = "".join(domain.dim(dim_idx).name for dim_idx in range(domain.ndim))
            yield dict(level=level, name=f"l_{level}.tdb", axes=axes, shape=array.shape)


def iter_pixel_depths_meta(
    compressors: Mapping[int, tiledb.Filter]
) -> Iterator[Tuple[int, int]]:
    for comp_level, compressor in compressors.items():
        level_pixel_depth = get_pixel_depth(compressor)
        yield (comp_level, level_pixel_depth)


def get_pixel_depth(compressor: tiledb.Filter) -> int:
    if not isinstance(compressor, tiledb.WebpFilter):
        return 1
    webp_format = compressor.input_format
    if webp_format in (WebpInputFormat.WEBP_RGB, WebpInputFormat.WEBP_BGR):
        return 3
    if webp_format in (WebpInputFormat.WEBP_RGBA, WebpInputFormat.WEBP_BGRA):
        return 4
    raise ValueError(f"Invalid WebpInputFormat: {compressor.input_format}")


def get_axes_translation(
    compressor: tiledb.Filter, axes: str
) -> Mapping[str, Sequence[str]]:
    if isinstance(compressor, tiledb.WebpFilter):
        return {"Y": ["Y"], "X": ["X", "C"]}

    return {axis: [axis] for axis in axes}


def iter_color(attr_type: np.dtype, channel_num: int = 3) -> Iterator[Dict[str, int]]:
    min_val = (
        np.iinfo(attr_type).min
        if np.issubdtype(attr_type, np.integer)
        else sys.float_info.min
    )
    max_val = (
        np.iinfo(attr_type).max
        if np.issubdtype(attr_type, np.integer)
        else sys.float_info.max
    )

    if channel_num == 1:
        yield {"red": max_val, "green": max_val, "blue": max_val, "alpha": max_val}
    elif channel_num == 3:
        yield {"red": max_val, "green": min_val, "blue": min_val, "alpha": max_val}
        yield {"red": min_val, "green": max_val, "blue": min_val, "alpha": max_val}
        yield {"red": min_val, "green": min_val, "blue": max_val, "alpha": max_val}

    while True:
        if np.issubdtype(attr_type, np.integer):
            red = np.random.randint(low=min_val, high=max_val, dtype=attr_type).item(0)
            green = np.random.randint(low=min_val, high=max_val, dtype=attr_type).item(
                0
            )
            blue = np.random.randint(low=min_val, high=max_val, dtype=attr_type).item(0)
        else:
            raise NotImplementedError(
                "RGB-Float and RGB-1-0 formats are not yet supported for this iterator"
            )
        yield {"red": red, "green": green, "blue": blue, "alpha": max_val}


def get_rgba(value: int) -> Dict[str, int]:
    color = {
        "red": (value & 0xFF000000) // 2**24,
        "green": (value & 0x00FF0000) // 2**16,
        "blue": (value & 0x0000FF00) // 2**8,
        "alpha": value & 0x000000FF,
    }

    return color


def get_decimal_from_rgba(color: Mapping[str, int]) -> int:
    """
    Convert an 8-bit RGBA color to a single signed integer value
    :param color: The color dictionary to convert.
        Each component should be between 0 and 255 inclusive (8-bit unsigned integer)
    :returns: A 32-bit signed integer in 2's complement representing the RGBA color
    """

    # Shift each 8-bit color component to the appropriate position
    # |  red   | green  |  blue  | alpha  |
    # |00000000|00000000|00000000|00000000| -> 32-bit integer
    decimal_color = (
        (color["red"] << 24)
        + (color["green"] << 16)
        + (color["blue"] << 8)
        + (color["alpha"])
    )

    # If the first bit is 1 then the binary representation in 2's complement should represent a negative value in decimal
    if decimal_color >> 31 == 1:
        # https://math.stackexchange.com/questions/285459/how-can-i-convert-2s-complement-to-decimal
        return -((~decimal_color & 0xFFFFFFFF) + 1)
    else:
        return decimal_color


def compute_channel_minmax(
    min_max: np.ndarray, tile_min: np.ndarray, tile_max: np.ndarray
) -> None:
    min_max[:, 0] = np.minimum(min_max[:, 0], tile_min)
    min_max[:, 1] = np.maximum(min_max[:, 1], tile_max)


def resolve_path(uri: str) -> Tuple[str, str]:
    parsed_uri = urlparse(uri)
    # normalize uri if it's a local path (e.g. ../..foo/bar)

    # Windows paths produce single letter scheme matching the drive letter
    # Unix absolute path produce an empty scheme
    resolved_uri = parsed_uri.path
    if is_win_path(parsed_uri.scheme):
        resolved_uri = str(Path(parsed_uri.path).resolve()).replace("\\", "/")
    return resolved_uri, parsed_uri.scheme


def is_win_path(scheme: str) -> bool:
    return len(scheme) < 2 or scheme == "file"


def is_local_path(scheme: str) -> bool:
    return True if is_win_path(scheme) or scheme == "" else False


def get_logger(level: int = logging.INFO, name: str = __name__) -> logging.Logger:
    """
    Get a logger with a custom formatter and set the logging level.

    :param level: logging level, defaults to logging.INFO
    :param name: logger name, defaults to __name__
    :return: Logger object
    """

    sh = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(module)s] [%(funcName)s] [%(levelname)s] %(message)s"
    )
    sh.setFormatter(formatter)

    logger = logging.getLogger(name)
    # Only add one handler, in case get_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(sh)
        logger.setLevel(level)

    return logger


def get_logger_wrapper(
    verbose: bool = False,
) -> logging.Logger:
    """
    Get a logger instance and log version information.

    :param verbose: verbose logging, defaults to False
    :return: logger instance
    """

    level = logging.DEBUG if verbose else logging.WARNING
    logger = get_logger(level)

    logger.debug(
        "tiledb=%s, libtiledb=%s, tiledb-bioimg=%s",
        tiledb.version(),
        tiledb.libtiledb.version(),
        version_tuple,
    )

    return logger
