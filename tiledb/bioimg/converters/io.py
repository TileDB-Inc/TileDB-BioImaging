import warnings
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from numpy._typing import NDArray
from tifffile import TiffPage
from tifffile.tifffile import (
    COMPRESSION,
    TiffFrame,
)


def as_array(
    page: Union[TiffPage, TiffFrame],
    buffer: NDArray[Any],
    indices: Sequence[int],
    order: Sequence[int],
    skip: int,
    length: int,
) -> NDArray[Any]:
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

    offset = "_ZYXS".index(page.axes[0])

    def decode(seg: Tuple[Optional[bytes], int]) -> None:
        data, start, size = keyframe.decode(*seg, **decodeargs)
        start = tuple(start[i + offset] for i in range(len(order)))
        size = tuple(size[i + offset - 1] for i in range(len(order)))

        index = tuple(slice(start[i], start[i] + size[i]) for i in range(len(size)))
        reshape = tuple(data.shape[i + offset - 1] for i in range(len(size)))

        buffer[index] = data.reshape(reshape)

    with ThreadPoolExecutor(4) as executor:
        for segment in fh.read_segments(
            itemgetter(*indices[skip : skip + length])(page.dataoffsets),
            itemgetter(*indices[skip : skip + length])(page.databytecounts),
            lock=lock,
            sort=True,
            flat=True,
            indices=indices[:length],
        ):
            executor.submit(decode, segment)
