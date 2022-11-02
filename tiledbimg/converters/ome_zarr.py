import glob
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import tiledb

# import xmltodict
import zarr
from numcodecs import Blosc
from ome_zarr.reader import Node, Reader, ZarrLocation
from ome_zarr.writer import write_multiscale

from .base import ImageConverter, ImageReader, ImageWriter


@dataclass(frozen=True)
class Level:
    node: Node


class OMEZarrWriter(ImageWriter):
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        # OME ZARR GROUP
        self.image_group = zarr.group(
            store=zarr.storage.DirectoryStore(path=self.output_path), overwrite=True
        )
        self.tldb_image = tiledb.Group(input_path, "r")
        # levels is a list of TLDB ARRAYS
        self._levels = sorted([resolution.uri for resolution in self.tldb_image])
        self.level_arrays = [lev_uri for lev_uri in self._levels]

    def uri(self, level: int) -> str:
        return os.path.join(self.input_path, f"{level}")

    @property
    def level_count(self) -> int:
        return len(self.tldb_image)

    def level_image(self, level: int) -> np.ndarray:
        with tiledb.open(self._levels[level]) as L:
            temp_array = L[:].swapaxes(0, 2)
            c, y, x = temp_array.shape
            tczyx_shape = (1, c, 1, y, x)
            return temp_array.reshape(tczyx_shape)

    def level_metadata(self, level: int) -> Dict[str, Any]:
        with tiledb.open(self._levels[level]) as L:
            return dict(pickle.loads(L.meta["pickled_zarrwriter_kwargs"]))

    def metadata(self) -> Dict[str, Any]:
        return dict(pickle.loads(self.tldb_image.meta["pickled_zarrwriter_kwargs"]))

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
                del zarray_meta.get("compressor")["id"]
                zarray_meta["compressor"] = Blosc.from_config(
                    zarray_meta.get("compressor")
                )
            level_metas_zarray.append(zarray_meta)

        # Write image does not support incremental pyramid write

        write_multiscale(
            images,
            self.image_group,
            scaler=None,
            axes=["t", "c", "z", "y", "x"],
            storage_options=level_metas_zarray,
            metadata=image_meta.get("metadata"),
        )


class OMEZarrReader(ImageReader):
    def __init__(self, input_path: str):
        self.image = Reader(ZarrLocation(input_path))
        resolutions = [
            img_path for img_path in glob.glob(f"{input_path}/[!OME]*", recursive=True)
        ]
        self.res_nodes = [
            nodes for res in resolutions for nodes in Reader(ZarrLocation(res))()
        ]
        self._levels = []
        for layer in self.res_nodes:
            self._levels.append(Level(layer))

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def level_image(self, level: int) -> np.ndarray:
        leveled_zarray = self._levels[level].node.data[0]
        # From NGFF format spec there is guarantee that axes are t,c,z,y,x
        return np.asarray(leveled_zarray).squeeze()

    def level_metadata(self, level: int) -> Dict[str, Any]:
        meta_root = self._levels[level].node.zarr
        writer_kwargs = dict(
            fmt=meta_root.fmt,
            root_attrs=meta_root.root_attrs,
            zarray=meta_root.zarray,
            zgroup=meta_root.zgroup,
        )
        return {"pickled_zarrwriter_kwargs": pickle.dumps(writer_kwargs)}

    def metadata(self) -> Dict[str, Any]:
        writer_kwargs = dict(
            bioformats2raw_layout=3,
            zarr_format=self.image.zarr.zgroup.get("zarr_format", 0),
            metadata=self.image.zarr.root_attrs.get("multiscales")[0].get("metadata"),
        )
        return {"pickled_zarrwriter_kwargs": pickle.dumps(writer_kwargs)}


class OMEZarrConverter(ImageConverter):
    """Converter of Zarr-supported images to TileDB Groups of Arrays"""

    def _get_image_reader(self, input_path: str) -> ImageReader:
        return OMEZarrReader(input_path)

    def _get_image_writer(self, input_path: str, output_path: str) -> ImageWriter:
        return OMEZarrWriter(input_path, output_path)
