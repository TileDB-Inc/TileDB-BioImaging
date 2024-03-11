import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy._typing import NDArray
from tifffile import TiffPage
from tifffile.tifffile import (
    NullContext,
    TiffFileError,
    TiffFrame,
    create_output,
    product,
)


def asarray(
    page: Union[TiffPage, TiffFrame],
    *,
    out: Optional[Union[str, BinaryIO, NDArray[Any]]] = None,
    squeeze: bool = True,
    lock: Optional[Union[threading.RLock, NullContext]] = None,
    maxworkers: Optional[int] = None,
    slices: Optional[Sequence[slice]] = None,
    storedTilesOrder: Optional[int] = None,
) -> NDArray[Any]:
    keyframe = page.keyframe  # self or keyframe

    if not keyframe.shaped or product(keyframe.shaped) == 0 or keyframe._dtype is None:
        return np.empty((0,), keyframe.dtype)

    if len(page.dataoffsets) == 0:
        raise TiffFileError("missing data offset")

    result_shaped = list(
        keyframe.shaped
    )  # [1, Z, Y, X, S] planar config 2 is not supported

    if slices:
        if storedTilesOrder == 1:
            size = int(np.prod(page.chunked[1:]))

            # For row major images only one slice is accepted
            assert len(slices) == 1

            if page.axes.startswith("Y"):
                result_shaped[2] = min(
                    int((slices[0].stop - slices[0].start) / size) * page.chunks[0],
                    result_shaped[2],
                )
            else:
                result_shaped[1] = min(
                    int((slices[0].stop - slices[0].start) / size) * page.chunks[0],
                    result_shaped[1],
                )
        elif storedTilesOrder == 2:
            result_shaped[3] = min(len(slices) * page.chunks[-2], result_shaped[3])

    result = create_output(out, tuple(result_shaped), keyframe.dtype)

    fh = page.parent.filehandle
    if lock is None:
        lock = fh.lock
        with lock:
            closed = fh.closed
            if closed:
                warnings.warn(f"{page!r} reading array from closed file", UserWarning)
                fh.open()
            keyframe.decode  # init TiffPage.decode function under lock

        def func(
            decoderesult: Tuple[
                Optional[NDArray[Any]],
                Tuple[int, int, int, int, int],
                Tuple[int, int, int, int],
            ],
            keyframe: TiffPage = keyframe,
            out: NDArray[Any] = result,
        ) -> None:
            # copy decoded segments to output array
            segment, (s, d, h, w, _), shape = decoderesult
            if segment is None:
                out[
                    s, d : d + shape[0], h : h + shape[1], w : w + shape[2]
                ] = keyframe.nodata
            else:
                out[s, d : d + shape[0], h : h + shape[1], w : w + shape[2]] = segment[
                    : keyframe.imagedepth - d,
                    : keyframe.imagelength - h,
                    : keyframe.imagewidth - w,
                ]
            # except IndexError:
            #     pass  # corrupted file, for example, with too many strips

        for _ in segments(
            page,
            func=func,
            lock=lock,
            maxworkers=maxworkers,
            sort=True,
            _fullsize=False,
            slices=slices,
        ):
            pass

    # result.shape = keyframe.

    # Squeeze array to the correct dimensionality
    if page.axes == "YX":
        result.shape = (result.shape[2], result.shape[3])
    elif page.axes == "YXS":
        result.shape = (result.shape[2], result.shape[3], result.shape[4])
    elif page.axes == "ZYX":
        result.shape = (result.shape[1], result.shape[2], result.shape[3])
    elif page.axes == "ZYXS":
        result.shape = (
            result.shape[1],
            result.shape[2],
            result.shape[3],
            result.shape[4],
        )
    else:
        raise ValueError(f"Unsupported axes {page.axes}")

    if closed:
        # TODO: close file if an exception occurred above
        fh.close()
    return result


def segments(
    page: TiffPage,
    *,
    lock: Optional[Union[threading.RLock, NullContext]] = None,
    maxworkers: Optional[int] = None,
    func: Optional[Callable[..., Any]] = None,  # TODO: type this
    sort: bool = False,
    _fullsize: Optional[bool] = None,
    slices: Optional[Sequence[slice]] = None,
) -> Iterator[
    Tuple[
        Optional[NDArray[Any]],
        Tuple[int, int, int, int, int],
        Tuple[int, int, int, int],
    ]
]:
    keyframe = page.keyframe  # self or keyframe
    fh = page.parent.filehandle
    if lock is None:
        lock = fh.lock
    if _fullsize is None:
        _fullsize = keyframe.is_tiled

    decodeargs: Dict[str, Any] = {"_fullsize": bool(_fullsize)}
    if keyframe.compression in {6, 7, 34892, 33007}:  # JPEG
        decodeargs["jpegtables"] = page.jpegtables
        decodeargs["jpegheader"] = keyframe.jpegheader

    if func is None:

        def decode(args, decodeargs=decodeargs, decode=keyframe.decode):  # type: ignore
            return decode(*args, **decodeargs)

    else:

        def decode(args, decodeargs=decodeargs, decode=keyframe.decode):  # type: ignore
            return func(decode(*args, **decodeargs))  # type: ignore

    dataoffsets = []
    databytecounts = []
    indices = None
    if slices:
        indices = []
        for i, sl in enumerate(slices):
            dataoffsets += page.dataoffsets[sl]
            databytecounts += page.databytecounts[sl]
            indices += [idx for idx in range(i, len(page.dataoffsets), sl.step)]

    else:
        dataoffsets = page.dataoffsets
        databytecounts = page.databytecounts

    if maxworkers is None or maxworkers < 1:
        maxworkers = keyframe.maxworkers
    if maxworkers < 2:  # type: ignore
        for segment in fh.read_segments(
            dataoffsets,
            databytecounts,
            lock=lock,
            sort=sort,
            flat=True,
            indices=indices,
        ):
            yield decode(segment)  # type: ignore
    else:
        # reduce memory overhead by processing chunks of up to
        # ~256 MB of segments because ThreadPoolExecutor.map is not
        # collecting iterables lazily
        with ThreadPoolExecutor(maxworkers) as executor:
            for segment in fh.read_segments(
                dataoffsets,
                databytecounts,
                lock=lock,
                sort=sort,
                flat=False,
                indices=indices,
            ):
                yield from executor.map(decode, segment)
