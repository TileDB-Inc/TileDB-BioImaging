import pickle
from typing import Any, Dict

import numpy as np
import tifffile

from .base import ImageConverter, ImageReader


class OMETiffReader(ImageReader):
    def __init__(self, input_path: str):
        self._tiff = tifffile.TiffFile(input_path)
        omexml = self._tiff.ome_metadata
        self._ome_metadata = tifffile.xml2dict(omexml) if omexml else {}
        self._levels = []
        self._subifds = {}
        for series in self._tiff.series:
            self._levels.extend(series.levels)
            self._subifds[series] = len(series.levels) - 1

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def level_image(self, level: int) -> np.ndarray:
        image = self._levels[level].asarray()
        assert image.ndim == 3
        # TODO: remove (hardcoded) swapaxes, need axes metadata
        image = image.swapaxes(0, 2)
        return image

    def level_metadata(self, level: int) -> Dict[str, Any]:
        series = self._levels[level]
        subifds = self._subifds.get(series)
        if subifds is not None:
            metadata = dict(self._ome_metadata, axes=series.axes)
        else:
            metadata = None
        keyframe = series.keyframe
        write_kwargs = dict(
            subifds=subifds,
            metadata=metadata,
            photometric=keyframe.photometric,
            planarconfig=keyframe.planarconfig,
            extrasamples=keyframe.extrasamples,
            rowsperstrip=keyframe.rowsperstrip,
            bitspersample=keyframe.bitspersample,
            compression=keyframe.compression,
            predictor=keyframe.predictor,
            subsampling=keyframe.subsampling,
            jpegtables=keyframe.jpegtables,
            colormap=keyframe.colormap,
            subfiletype=keyframe.subfiletype or None,
            software=keyframe.software,
            tile=keyframe.tile,
            datetime=keyframe.datetime,
            resolution=keyframe.resolution,
            resolutionunit=keyframe.resolutionunit,
        )
        return {"pickled_write_kwargs": pickle.dumps(write_kwargs)}

    def metadata(self) -> Dict[str, Any]:
        writer_kwargs = dict(
            bigtiff=self._tiff.is_bigtiff,
            byteorder=self._tiff.byteorder,
            append=self._tiff.is_appendable,
            imagej=self._tiff.is_imagej,
            ome=self._tiff.is_ome,
        )
        return {"pickled_tiffwriter_kwargs": pickle.dumps(writer_kwargs)}


class OMETiffConverter(ImageConverter):
    """Converter of Tiff-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OMETiffReader(input_path)
