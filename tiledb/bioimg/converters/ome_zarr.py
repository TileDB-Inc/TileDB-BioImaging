from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.reader import OMERO, Multiscales, Reader, ZarrLocation
from ome_zarr.writer import write_multiscale

from tiledb.cc import WebpInputFormat

from .axes import Axes
from .base import ImageConverter, ImageReader, ImageWriter


class OMEZarrReader(ImageReader):
    def __init__(self, input_path: str):
        """
        OME-Zarr image reader

        :param input_path: The path to the Zarr image
        """
        root_node = next(Reader(ZarrLocation(input_path))())
        self._multiscales = cast(Multiscales, root_node.load(Multiscales))
        self._omero = cast(Optional[OMERO], root_node.load(OMERO))

    @property
    def axes(self) -> Axes:
        return Axes(a["name"].upper() for a in self._multiscales.node.metadata["axes"])

    @property
    def channels(self) -> Sequence[str]:
        return tuple(self._omero.node.metadata.get("name", ())) if self._omero else ()

    @property
    def webp_format(self) -> WebpInputFormat:
        channels = self._omero.image_data.get("channels", ()) if self._omero else ()
        colors = tuple(channel.get("color") for channel in channels)
        if colors == ("FF0000", "00FF00", "0000FF"):
            return WebpInputFormat.WEBP_RGB
        return WebpInputFormat.WEBP_NONE

    @property
    def level_count(self) -> int:
        return len(self._multiscales.datasets)

    def level_dtype(self, level: int) -> np.dtype:
        return self._multiscales.node.data[level].dtype

    def level_shape(self, level: int) -> Tuple[int, ...]:
        return cast(Tuple[int, ...], self._multiscales.node.data[level].shape)

    def level_image(
        self, level: int, tile: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        dask_array = self._multiscales.node.data[level]
        if tile is not None:
            dask_array = dask_array[tile]
        return np.asarray(dask_array)

    def level_metadata(self, level: int) -> Dict[str, Any]:
        dataset = self._multiscales.datasets[level]
        location = ZarrLocation(self._multiscales.zarr.subpath(dataset))
        return {"json_zarray": json.dumps(location.zarray)}

    @property
    def group_metadata(self) -> Dict[str, Any]:
        multiscale = self._multiscales.lookup("multiscales", [])[0]
        writer_kwargs = dict(
            axes=multiscale.get("axes"),
            coordinate_transformations=[
                d.get("coordinateTransformations") for d in multiscale["datasets"]
            ],
            name=multiscale.get("name"),
            metadata=multiscale.get("metadata"),
            omero=self._omero.image_data if self._omero else None,
        )
        return {"json_zarrwriter_kwargs": json.dumps(writer_kwargs)}


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
