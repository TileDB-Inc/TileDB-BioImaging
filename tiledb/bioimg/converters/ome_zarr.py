from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, cast

import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.reader import Reader, ZarrLocation
from ome_zarr.writer import write_multiscale

from .base import Axes, ImageConverter, ImageReader, ImageWriter


class OMEZarrReader(ImageReader):
    def __init__(self, input_path: str):
        """
        OME-Zarr image reader

        :param input_path: The path to the Zarr image
        """
        self._root_attrs = ZarrLocation(input_path).root_attrs
        self._nodes = []
        for dataset in self._multiscale["datasets"]:
            path = os.path.join(input_path, dataset["path"])
            self._nodes.extend(Reader(ZarrLocation(path))())

    @property
    def axes(self) -> Axes:
        return Axes(axis["name"].upper() for axis in self._multiscale["axes"])

    @property
    def level_count(self) -> int:
        return len(self._nodes)

    def level_image(self, level: int) -> np.ndarray:
        data = self._nodes[level].data
        assert len(data) == 1
        return np.asarray(data[0])

    def level_metadata(self, level: int) -> Dict[str, Any]:
        return {"json_zarray": json.dumps(self._nodes[level].zarr.zarray)}

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
            omero=self._root_attrs.get("omero"),
        )
        return {"json_zarrwriter_kwargs": json.dumps(writer_kwargs)}

    @property
    def _multiscale(self) -> Dict[str, Any]:
        multiscales = self._root_attrs["multiscales"]
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

    def write_group_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._group_metadata = json.loads(metadata["json_zarrwriter_kwargs"])

    def write_level_image(
        self, level: int, image: np.ndarray, metadata: Mapping[str, Any]
    ) -> None:
        # store the image to be written at __exit__
        self._pyramid.append(image)
        # store the zarray metadata to be written at __exit__
        zarray = json.loads(metadata["json_zarray"])
        compressor = zarray["compressor"]
        del compressor["id"]
        zarray["compressor"] = Blosc.from_config(compressor)
        self._storage_options.append(zarray)

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

    _ImageReaderType = OMEZarrReader
    _ImageWriterType = OMEZarrWriter
