from __future__ import annotations

import json
import os
from typing import Any, Dict, List, cast

import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.reader import Reader, ZarrLocation
from ome_zarr.writer import write_multiscale

import tiledb

from .base import Axes, ImageConverter, ImageReader, ImageWriter


class OMEZarrReader(ImageReader):
    def __init__(self, input_path: str):
        """
        OME-Zarr image reader

        :param input_path: The path to the Zarr image

        """
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
        writer_kwargs = dict(
            axes=multiscale.get("axes"),
            coordinate_transformations=[
                d.get("coordinateTransformations") for d in multiscale["datasets"]
            ],
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


class OMEZarrWriter(ImageWriter):
    def __init__(self, output_path: str):
        """
        OME-Zarr image writer from TileDB

        :param output_path: The path to the Zarr image

        """
        self._group = zarr.group(
            store=zarr.storage.DirectoryStore(path=output_path), overwrite=True
        )
        self._pyramid: List[np.ndarray] = []
        self._storage_options: List[Dict[str, Any]] = []
        self._group_metadata: Dict[str, Any] = {}

    def write_level_array(self, level: int, array: tiledb.Array) -> None:
        # store the image to be written at __exit__
        image = array[:]
        c, y, x = image.shape
        tczyx_shape = (1, c, 1, y, x)
        self._pyramid.append(image.reshape(tczyx_shape))

        # store the zarray metadata to be written at __exit__
        zarray = json.loads(array.meta["json_zarray"])
        compressor = zarray["compressor"]
        del compressor["id"]
        zarray["compressor"] = Blosc.from_config(compressor)
        self._storage_options.append(zarray)

    def write_group_metadata(self, group: tiledb.Group) -> None:
        self._group_metadata = json.loads(group.meta["json_zarrwriter_kwargs"])

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        group_metadata = self._group_metadata
        write_multiscale(
            pyramid=self._pyramid,
            group=self._group,
            axes=group_metadata["axes"],
            coordinate_transformations=group_metadata["coordinate_transformations"],
            storage_options=self._storage_options,
            name=group_metadata["name"],
            metadata=group_metadata["metadata"],
        )
        if group_metadata["omero"]:
            self._group.attrs["omero"] = group_metadata["omero"]


class OMEZarrConverter(ImageConverter):
    """Converter of Zarr-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OMEZarrReader(input_path)

    def _get_image_writer(self, output_path: str) -> ImageWriter:
        return OMEZarrWriter(output_path)
