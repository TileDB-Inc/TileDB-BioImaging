from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Sequence, cast

import numpy as np
import tiledb
import zarr
from numcodecs import Blosc
from ome_zarr.reader import Reader, ZarrLocation
from ome_zarr.writer import write_multiscale

from .base import Axes, ImageConverter, ImageReader, ImageWriter


class OMEZarrWriter(ImageWriter):
    def __init__(self, input_path: str, output_path: str):
        # OME ZARR GROUP
        self._output_group = zarr.group(
            store=zarr.storage.DirectoryStore(path=output_path), overwrite=True
        )
        self._input_group = tiledb.Group(input_path, "r")

        # levels is a list of TLDB ARRAYS
        self._levels = {}
        for resolution in self._input_group:
            uri = resolution.uri
            with tiledb.open(uri) as a:
                level = a.meta.get("level", 0)
            self._levels.update({level: uri})

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def level_image(self, level: int) -> np.ndarray:
        with tiledb.open(self._levels.get(level)) as L:
            data = L[:]
            c, y, x = data.shape
            tczyx_shape = (1, c, 1, y, x)
            return data.reshape(tczyx_shape)

    def level_metadata(self, level: int) -> Dict[str, Any]:
        with tiledb.open(self._levels.get(level)) as L:
            return cast(
                Dict[str, Any], pickle.loads(L.meta["pickled_zarrwriter_kwargs"])
            )

    def metadata(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            pickle.loads(self._input_group.meta["pickled_zarrwriter_kwargs"]),
        )

    def write(
        self,
        images: Sequence[np.ndarray],
        level_metas: Sequence[Dict[str, Any]],
        image_meta: Dict[str, Any] = {},
    ) -> None:
        level_metas_zarray = []
        for level_meta in level_metas:
            zarray_meta = level_meta.get("zarray")
            if zarray_meta is not None:
                compressor = zarray_meta["compressor"]
                del compressor["id"]
                zarray_meta["compressor"] = Blosc.from_config(compressor)
                level_metas_zarray.append(zarray_meta)

        # Write image does not support incremental pyramid write
        write_multiscale(
            images,
            group=self._output_group,
            axes=image_meta["axes"],
            coordinate_transformations=image_meta["coordinate_transformations"],
            storage_options=level_metas_zarray,
            name=image_meta["name"],
            metadata=image_meta["metadata"],
        )
        if image_meta["omero"]:
            self._output_group.attrs["omero"] = image_meta["omero"]


class OMEZarrReader(ImageReader):
    def __init__(self, input_path: str):
        self.root_attrs = ZarrLocation(input_path).root_attrs
        self.nodes = []
        for dataset in self._multiscale["datasets"]:
            path = os.path.join(input_path, dataset["path"])
            self.nodes.extend(Reader(ZarrLocation(path))())

    @property
    def level_count(self) -> int:
        return len(self.nodes)

    def level_axes(self, level: int) -> Axes:
        return Axes("CYX")

    def level_image(self, level: int) -> np.ndarray:
        data = self.nodes[level].data
        assert len(data) == 1
        leveled_zarray = data[0]
        if leveled_zarray.shape[0] != 1:
            raise NotImplementedError("T axes not supported yet")
        if leveled_zarray.shape[2] != 1:
            raise NotImplementedError("Z axes not supported yet")
        # From NGFF format spec there is guarantee that axes are t,c,z,y,x
        return np.asarray(data[0]).squeeze()

    def level_metadata(self, level: int) -> Dict[str, Any]:
        writer_kwargs = dict(zarray=self.nodes[level].zarr.zarray)
        return {"pickled_zarrwriter_kwargs": pickle.dumps(writer_kwargs)}

    def metadata(self) -> Dict[str, Any]:
        multiscale = self._multiscale
        coordinate_transformations = (
            d.get("coordinateTransformations") for d in multiscale["datasets"]
        )
        writer_kwargs = dict(
            axes=multiscale.get("axes"),
            coordinate_transformations=list(filter(None, coordinate_transformations)),
            name=multiscale.get("name"),
            metadata=multiscale.get("metadata"),
            omero=self.root_attrs.get("omero"),
        )
        return {"pickled_zarrwriter_kwargs": pickle.dumps(writer_kwargs)}

    @property
    def _multiscale(self) -> Dict[str, Any]:
        multiscales = self.root_attrs["multiscales"]
        assert len(multiscales) == 1, multiscales
        return cast(Dict[str, Any], multiscales[0])


class OMEZarrConverter(ImageConverter):
    """Converter of Zarr-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OMEZarrReader(input_path)

    def _get_image_writer(self, input_path: str, output_path: str) -> ImageWriter:
        return OMEZarrWriter(input_path, output_path)
