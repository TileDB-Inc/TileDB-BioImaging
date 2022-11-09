from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence, cast

import numpy as np
import tiledb
import zarr
from numcodecs import Blosc
from ome_zarr.reader import Reader, ZarrLocation
from ome_zarr.writer import write_multiscale

from .base import Axes, ImageConverter, ImageReader, ImageWriter


class OMEZarrWriter(ImageWriter):
    def __init__(self, output_path: str):
        self._output_group = zarr.group(
            store=zarr.storage.DirectoryStore(path=output_path), overwrite=True
        )

    def level_image(self, array: tiledb.Array) -> np.ndarray:
        data = array[:]
        c, y, x = data.shape
        tczyx_shape = (1, c, 1, y, x)
        return data.reshape(tczyx_shape)

    def level_metadata(self, array: tiledb.Array) -> Dict[str, Any]:
        zarray = json.loads(array.meta["json_zarray"])
        compressor = zarray["compressor"]
        del compressor["id"]
        zarray["compressor"] = Blosc.from_config(compressor)
        return cast(Dict[str, Any], zarray)

    def group_metadata(self, group: tiledb.Group) -> Dict[str, Any]:
        return cast(Dict[str, Any], json.loads(group.meta["json_zarrwriter_kwargs"]))

    def write(
        self,
        images: Sequence[np.ndarray],
        level_metadata: Sequence[Dict[str, Any]],
        group_metadata: Dict[str, Any],
    ) -> None:
        # Write image does not support incremental pyramid write
        write_multiscale(
            list(images),
            group=self._output_group,
            axes=group_metadata["axes"],
            coordinate_transformations=group_metadata["coordinate_transformations"],
            storage_options=list(level_metadata),
            name=group_metadata["name"],
            metadata=group_metadata["metadata"],
        )
        if group_metadata["omero"]:
            self._output_group.attrs["omero"] = group_metadata["omero"]


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
        return {"json_zarray": json.dumps(self.nodes[level].zarr.zarray)}

    @property
    def group_metadata(self) -> Dict[str, Any]:
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
        return {"json_zarrwriter_kwargs": json.dumps(writer_kwargs)}

    @property
    def _multiscale(self) -> Dict[str, Any]:
        multiscales = self.root_attrs["multiscales"]
        assert len(multiscales) == 1, multiscales
        return cast(Dict[str, Any], multiscales[0])


class OMEZarrConverter(ImageConverter):
    """Converter of Zarr-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OMEZarrReader(input_path)

    def _get_image_writer(self, output_path: str) -> ImageWriter:
        return OMEZarrWriter(output_path)
