import pickle
from typing import Any, Dict

import numpy as np
import tifffile

from .base import ImageConverter, ImageReader


class OMETiffReader(ImageReader):
    def __init__(self, input_path: str):
        self._tiff = tifffile.TiffFile(input_path)
        self._ome_metadata = tifffile.xml2dict(self._tiff.ome_metadata)
        self._pages = []
        self._page_subifds = {}
        for s in self._tiff.series:
            for i, l in enumerate(s.levels):
                page = l.keyframe
                self._pages.append(page)
                if i == 0:
                    self._page_subifds[page] = len(s.levels) - 1

    @property
    def level_count(self) -> int:
        return len(self._pages)

    def level_image(self, level: int) -> np.ndarray:
        image = self._pages[level].asarray()
        if image.ndim == 3:
            # TODO: remove (hardcoded) swapaxes, need axes metadata
            image = image.swapaxes(0, 2)
        return image

    def level_metadata(self, level: int) -> Dict[str, Any]:
        page = self._pages[level]
        subifds = self._page_subifds.get(page)
        if subifds is not None:
            metadata = dict(self._ome_metadata, axes=page.axes)
        else:
            metadata = None
        write_kwargs = dict(
            subifds=subifds,
            metadata=metadata,
            photometric=page.photometric,
            planarconfig=page.planarconfig,
            extrasamples=page.extrasamples,
            rowsperstrip=page.rowsperstrip,
            bitspersample=page.bitspersample,
            compression=page.compression,
            predictor=page.predictor,
            subsampling=page.subsampling,
            jpegtables=page.jpegtables,
            colormap=page.colormap,
            subfiletype=page.subfiletype or None,
            software=page.software,
            tile=page.tile,
            datetime=page.datetime,
            resolution=page.resolution,
            resolutionunit=page.resolutionunit,
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
