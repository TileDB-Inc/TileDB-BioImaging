import math
import warnings
from logging import Logger
from typing import (
    Any,
    Dict,
    Iterator,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy
from numpy._typing import NDArray
from tifffile import TiffPage
from tifffile.tifffile import (
    COMPRESSION,
    TiffFrame,
)

# TODO: Add support for 3D images


def as_array(
    page: Union[TiffPage, TiffFrame], logger: Logger, buffer_size: Optional[int] = None
) -> Iterator[Tuple[NDArray[Any], Tuple[int, ...]]]:
    keyframe = page.keyframe  # self or keyframe
    fh = page.parent.filehandle
    lock = fh.lock
    with lock:
        closed = fh.closed
        if closed:
            warnings.warn(f"{page!r} reading array from closed file", UserWarning)
            fh.open()
        keyframe.decode  # init TiffPage.decode function under lock

    decodeargs: Dict[str, Any] = {"_fullsize": bool(False)}
    if keyframe.compression in {
        COMPRESSION.OJPEG,
        COMPRESSION.JPEG,
        COMPRESSION.JPEG_LOSSY,
        COMPRESSION.ALT_JPEG,
    }:  # JPEG
        decodeargs["jpegtables"] = page.jpegtables
        decodeargs["jpegheader"] = keyframe.jpegheader

    segment_cache: MutableSequence[Optional[Tuple[Optional[bytes], int]]] = [
        None
    ] * math.prod(page.chunked)

    x_chunks = page.chunked[page.axes.index("X")] if "X" in page.axes else 1
    y_chunks = page.chunked[page.axes.index("Y")] if "Y" in page.axes else 1
    z_chunks = page.chunked[page.axes.index("Z")] if "Z" in page.axes else 1

    if keyframe.is_tiled:
        tilewidth = keyframe.tilewidth
        tilelength = keyframe.tilelength
        tiledepth = keyframe.tiledepth
    else:
        # striped image
        tilewidth = keyframe.imagewidth
        tilelength = keyframe.rowsperstrip
        tiledepth = 1  # TODO: Find 3D striped image to test

    for segment in fh.read_segments(
        page.dataoffsets, page.databytecounts, lock=lock, buffersize=buffer_size
    ):
        x_index = segment[1] % x_chunks
        y_index = (segment[1] // x_chunks) % y_chunks
        z_index = segment[1] // (y_chunks * x_chunks)

        if z_index >= z_chunks or y_index >= y_chunks or x_index >= x_chunks:
            logger.warning(
                f"Found segment with index (Z: {z_index},Y: {y_index}, X:{x_index}) outside of bounds. Check for source file corruption."
            )
            continue

        segment_cache[segment[1]] = segment

        while (offsets := has_row(segment_cache, page)) is not None:
            y_offset, z_offset = offsets

            x_size = keyframe.imagewidth
            y_size = min(
                keyframe.imagelength - y_offset * tilelength,
                tilelength,
            )
            z_size = min(keyframe.imagedepth - z_offset * tiledepth, tiledepth)

            buffer = numpy.zeros(
                shape=(1, z_size, y_size, x_size, keyframe.samplesperpixel),
                dtype=page.dtype,
            )

            for x_offset in range(x_chunks):
                idx = z_offset * y_chunks * x_chunks + y_offset * x_chunks + x_offset
                data, (_, z, y, x, s), size = keyframe.decode(
                    *segment_cache[idx], **decodeargs
                )
                buffer[0, 0 : size[0], 0 : size[1], x : x + size[2]] = data[
                    : keyframe.imagedepth - z,
                    : keyframe.imagelength - y,
                    : keyframe.imagewidth - x,
                ]

                segment_cache[idx] = None

            shape = (z_size,) if "Z" in page.axes else ()
            shape = shape + (y_size,) if "Y" in page.axes else shape
            shape = shape + (x_size,) if "X" in page.axes else shape
            shape = shape + (keyframe.samplesperpixel,) if "S" in page.axes else shape

            offset = (z_offset * tiledepth,) if "Z" in page.axes else ()
            offset = offset + (y_offset * tilelength,) if "Y" in page.axes else offset
            offset = offset + (0,) if "X" in page.axes else offset
            offset = offset + (0,) if "S" in page.axes else offset

            yield buffer.reshape(shape), offset

        while (offsets := has_column(segment_cache, page)) is not None:
            x_offset, z_offset = offsets

            x_size = min(keyframe.imagewidth - x_offset * tilewidth, tilewidth)
            y_size = keyframe.imagelength
            z_size = min(keyframe.imagedepth - z_offset * tiledepth, tiledepth)

            buffer = numpy.zeros(
                shape=(1, z_size, y_size, x_size, keyframe.samplesperpixel),
                dtype=page.dtype,
            )

            for y_offset in range(y_chunks):
                idx = z_offset * y_chunks * x_chunks + y_offset * x_chunks + x_offset
                data, (_, z, y, x, s), size = keyframe.decode(
                    *segment_cache[idx], **decodeargs
                )
                buffer[0, 0 : size[0], y : y + size[1], 0 : size[2]] = data[
                    : keyframe.imagedepth - z,
                    : keyframe.imagelength - y,
                    : keyframe.imagewidth - x,
                ]

                segment_cache[idx] = None

            shape = (z_size,) if "Z" in page.axes else ()
            shape = shape + (y_size,) if "Y" in page.axes else shape
            shape = shape + (x_size,) if "X" in page.axes else shape
            shape = shape + (keyframe.samplesperpixel,) if "S" in page.axes else shape

            offset = (z_offset * tiledepth,) if "Z" in page.axes else ()
            offset = offset + (0,) if "Y" in page.axes else offset
            offset = offset + (x_offset * tilewidth,) if "X" in page.axes else offset
            offset = offset + (0,) if "S" in page.axes else offset

            yield buffer.reshape(shape), offset

        while (offsets := has_depth(segment_cache, page)) is not None:
            x_offset, y_offset = offsets

            x_size = min(keyframe.imagewidth - x_offset * tilewidth, tilewidth)
            y_size = min(keyframe.imagelength - y_offset * tilelength, tilelength)
            z_size = keyframe.imagedepth

            buffer = numpy.zeros(
                shape=(1, z_size, y_size, x_size, keyframe.samplesperpixel),
                dtype=page.dtype,
            )

            for z_offset in range(z_chunks):
                idx = z_offset * y_chunks * x_chunks + y_offset * x_chunks + x_offset
                data, (_, z, y, x, s), size = keyframe.decode(
                    *segment_cache[idx], **decodeargs
                )
                buffer[0, 0 : size[0], y : y + size[1], 0 : size[2]] = data[
                    : keyframe.imagedepth - z,
                    : keyframe.imagelength - y,
                    : keyframe.imagewidth - x,
                ]

                segment_cache[idx] = None

            shape = (z_size,) if "Z" in page.axes else ()
            shape = shape + (y_size,) if "Y" in page.axes else shape
            shape = shape + (x_size,) if "X" in page.axes else shape
            shape = shape + (keyframe.samplesperpixel,) if "S" in page.axes else shape

            offset = (0,) if "Z" in page.axes else ()
            offset = offset + (y_offset * tilelength,) if "Y" in page.axes else offset
            offset = offset + (x_offset * tilewidth,) if "X" in page.axes else offset
            offset = offset + (0,) if "S" in page.axes else offset

            yield buffer.reshape(shape), offset


def has_row(
    segments: Sequence[Optional[Tuple[Optional[bytes], int]]], page: TiffPage
) -> Optional[Tuple[int, int]]:
    # TODO: Check bitarray for performance improvement

    x_chunks = page.chunked[page.axes.index("X")] if "X" in page.axes else 1
    y_chunks = page.chunked[page.axes.index("Y")] if "Y" in page.axes else 1
    z_chunks = page.chunked[page.axes.index("Z")] if "Z" in page.axes else 1

    for z_offset in range(z_chunks):
        for y_offset in range(y_chunks):
            for x_offset in range(x_chunks):
                idx = z_offset * x_chunks * y_chunks + y_offset * x_chunks + x_offset

                if segments[idx] is None:
                    break
            else:
                return y_offset, z_offset

    return None


def has_column(
    segments: Sequence[Optional[Tuple[Optional[bytes], int]]], page: TiffPage
) -> Optional[Tuple[int, int]]:
    # TODO: Check bitarray for performance improvement

    x_chunks = page.chunked[page.axes.index("X")] if "X" in page.axes else 1
    y_chunks = page.chunked[page.axes.index("Y")] if "Y" in page.axes else 1
    z_chunks = page.chunked[page.axes.index("Z")] if "Z" in page.axes else 1

    for z_offset in range(z_chunks):
        for x_offset in range(x_chunks):
            for y_offset in range(y_chunks):
                idx = z_offset * x_chunks * y_chunks + y_offset * x_chunks + x_offset

                if segments[idx] is None:
                    break
            else:
                return x_offset, z_offset

    return None


def has_depth(
    segments: Sequence[Optional[Tuple[Optional[bytes], int]]], page: TiffPage
) -> Optional[Tuple[int, int]]:
    # TODO: Check bitarray for performance improvement

    x_chunks = page.chunked[page.axes.index("X")] if "X" in page.axes else 1
    y_chunks = page.chunked[page.axes.index("Y")] if "Y" in page.axes else 1
    z_chunks = page.chunked[page.axes.index("Z")] if "Z" in page.axes else 1

    for y_offset in range(y_chunks):
        for x_offset in range(x_chunks):
            for z_offset in range(z_chunks):
                idx = z_offset * x_chunks * y_chunks + y_offset * x_chunks + x_offset

                if segments[idx] is None:
                    break
            else:
                return x_offset, y_offset

    return None
