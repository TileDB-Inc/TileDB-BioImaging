import math
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Any, Tuple, Iterator, BinaryIO

import imagecodecs
import numpy
from numpy._typing import NDArray
from tifffile import TiffPage
from tifffile.tifffile import NullContext, product, TiffFileError, create_output, TIFF, FileHandle
from typing_extensions import Optional, Union, Callable


def asarray(
        page: TiffPage,
        *,
        out:  Optional[Union[str, BinaryIO, NDArray[Any]]] = None,
        squeeze: bool = True,
        lock: Optional[Union[threading.RLock, NullContext]] = None,
        maxworkers: Optional[int] = None,
        slices: Optional[Sequence[slice]] = None,
        storedTilesOrder: Optional[int] = None
) -> NDArray[Any]:
    keyframe = page.keyframe  # self or keyframe

    if (
            not keyframe.shaped
            or product(keyframe.shaped) == 0
            or keyframe._dtype is None
    ):
        return numpy.empty((0,), keyframe.dtype)

    if len(page.dataoffsets) == 0:
        raise TiffFileError('missing data offset')

    fh = page.parent.filehandle
    if lock is None:
        lock = fh.lock

    if (
            isinstance(out, str)
            and out == 'memmap'
            and keyframe.is_memmappable
    ):
        # direct memory map array in file
        with lock:
            closed = fh.closed
            if closed:
                warnings.warn(
                    f'{page!r} reading array from closed file', UserWarning
                )
                fh.open()
            result = fh.memmap_array(
                keyframe.parent.byteorder + keyframe._dtype.char,
                keyframe.shaped,
                offset=page.dataoffsets[0],
            )

    elif keyframe.is_contiguous:
        # read contiguous bytes to array
        if keyframe.is_subsampled:
            raise NotImplementedError('chroma subsampling not supported')
        if out is not None:
            out = create_output(out, keyframe.shaped, keyframe._dtype)
        with lock:
            closed = fh.closed
            if closed:
                warnings.warn(
                    f'{page!r} reading array from closed file', UserWarning
                )
                fh.open()
            fh.seek(page.dataoffsets[0])
            result = fh.read_array(
                keyframe.parent.byteorder + keyframe._dtype.char,
                product(keyframe.shaped),
                out=out,
            )
        if keyframe.fillorder == 2:
            result = imagecodecs.bitorder_decode(result, out=result)
        if keyframe.predictor != 1:
            # predictors without compression
            unpredict = TIFF.UNPREDICTORS[keyframe.predictor]
            if keyframe.predictor == 1:
                result = unpredict(result, axis=-2, out=result)
            else:
                # floatpred cannot decode in-place
                out = unpredict(result, axis=-2, out=result)
                result[:] = out

    elif (
            keyframe.jpegheader is not None
            and keyframe is page
            and 273 in page.tags  # striped ...
            and page.is_tiled  # but reported as tiled
            # TODO: imagecodecs can decode larger JPEG
            and page.imagewidth <= 65500
            and page.imagelength <= 65500
    ):
        # decode the whole NDPI JPEG strip
        with lock:
            closed = fh.closed
            if closed:
                warnings.warn(
                    f'{page!r} reading array from closed file', UserWarning
                )
                fh.open()
            fh.seek(page.tags[273].value[0])  # StripOffsets
            data = fh.read(page.tags[279].value[0])  # StripByteCounts
        decompress = TIFF.DECOMPRESSORS[page.compression]
        result = decompress(
            data,
            bitspersample=page.bitspersample,
            out=out,
            # shape=(self.imagelength, self.imagewidth)
        )
        del data

    else:
        # decode individual strips or tiles
        with lock:
            closed = fh.closed
            if closed:
                warnings.warn(
                    f'{page!r} reading array from closed file', UserWarning
                )
                fh.open()
            keyframe.decode  # init TiffPage.decode function under lock

        result_shaped = list(keyframe.shaped)  # [1, Z, Y, X, S] planar config 2 is not supported

        if slices:
            if storedTilesOrder == 1:
                size = math.prod(page.chunked[1:])

                # For row major images only one slice is accepted
                assert len(slices) == 1

                if page.axes == "YXS":
                    result_shaped[2] = min(int((slices[0].stop - slices[0].start) / size) * page.chunks[0], result_shaped[2])
                elif page.axes == "ZYXS":
                    pass
            elif storedTilesOrder == 2:
                if page.axes == "YXS":
                    result_shaped[3] = min(len(slices) * page.chunks[-2], result_shaped[3])
                elif page.axes == "ZYXS":
                    pass

        result = create_output(out, tuple(result_shaped), keyframe._dtype)

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
                s, d: d + shape[0], h: h + shape[1], w: w + shape[2]
                ] = keyframe.nodata
            else:
                out[
                s, d: d + shape[0], h: h + shape[1], w: w + shape[2]
                ] = segment[
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
                slices=slices
        ):
            pass

    # result.shape = keyframe.shaped
    if squeeze:
        try:
            # result.shape = keyframe.shape
            result.shape = tuple(dim for dim in result.shape if dim != 1)
        except ValueError:
            warnings.warn(
                f'{page!r} '
                f'failed to reshape {result.shape} to {keyframe.shape}'
            )

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
        slices: Optional[Sequence[slice]] = None) -> Iterator[Tuple[Optional[NDArray[Any]], Tuple[int, int, int, int, int], Tuple[int, int, int, int],]]:
    keyframe = page.keyframe  # self or keyframe
    fh = page.parent.filehandle
    if lock is None:
        lock = fh.lock
    if _fullsize is None:
        _fullsize = keyframe.is_tiled

    decodeargs: dict[str, Any] = {'_fullsize': bool(_fullsize)}
    if keyframe.compression in {6, 7, 34892, 33007}:  # JPEG
        decodeargs['jpegtables'] = page.jpegtables
        decodeargs['jpegheader'] = keyframe.jpegheader

    if func is None:

        def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
            return decode(*args, **decodeargs)

    else:

        def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
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
    if maxworkers < 2:
        for segment in fh.read_segments(
                dataoffsets,
                databytecounts,
                lock=lock,
                sort=sort,
                flat=True,
                indices=indices,
        ):
            yield decode(segment)
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
                    indices=indices
            ):
                yield from executor.map(decode, segment)
