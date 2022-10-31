import pickle
from typing import Any, Dict, Optional

import numpy as np
import tifffile

from .base import Axes, ImageConverter, ImageReader, ImageWriter


class OMETiffReader(ImageReader):
    def __init__(self, input_path: str):
        self._tiff = tifffile.TiffFile(input_path)
        omexml = self._tiff.ome_metadata
        self._ome_metadata = tifffile.xml2dict(omexml) if omexml else {}
        # XXX ignore all but the first series
        self._levels = self._tiff.series[0].levels

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def level_axes(self, level: int) -> Axes:
        return Axes(self._levels[level].axes.replace("S", "C"))

    def level_image(self, level: int) -> np.ndarray:
        return self._levels[level].asarray()

    def level_metadata(self, level: int) -> Dict[str, Any]:
        series = self._levels[level]
        if level == 0:
            subifds: Optional[int] = len(series.levels) - 1
            metadata = dict(self._ome_metadata, axes=series.axes)
        else:
            subifds = metadata = None
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

    def _get_image_writer(self, input_path: str, output_path: str) -> ImageWriter:
        pass
