import json
from pathlib import Path

import numpy as np
import PIL.Image
import tiledb
import zarr

from tests import get_CMU_1_SMALL_REGION_schemas, get_path
from tiledbimg.converters.ome_zarr import OMEZarrConverter
from tiledbimg.openslide import LevelInfo, TileDBOpenSlide


def test_ome_zarr_converter(tmp_path):
    input_path = Path(get_path("CMU-1-Small-Region.ngff.zarr")) / "0.zarr"
    output_path = tmp_path / input_path.name
    OMEZarrConverter().to_tiledb(input_path, str(output_path))

    schema = get_CMU_1_SMALL_REGION_schemas()[0]
    assert len(tiledb.Group(str(output_path))) == 1
    with tiledb.open(str(output_path / f"l_{0}.tdb")) as A:
        assert A.schema == schema

    expected_dim = (2220, 2967)
    expected_downsample = (1.0,)
    t = TileDBOpenSlide.from_group_uri(str(output_path))
    assert t.level_count == 1
    assert t.dimensions == expected_dim
    assert t.level_downsamples == expected_downsample
    assert t.level_info[0] == LevelInfo(uri="", dimensions=schema.shape[:-3:-1])
    region = t.read_region(level=0, location=(100, 100), size=(300, 400))
    assert isinstance(region, np.ndarray)
    assert region.ndim == 3
    assert region.dtype == np.uint8
    img = PIL.Image.fromarray(region)
    assert img.size == (300, 400)


def test_ome_zarr_converter_images(tmp_path):
    OMEZarrConverter().convert_images(
        Path(get_path("CMU-1-Small-Region.ngff.zarr")).glob("*"),
        tmp_path,
        level_min=0,
        max_workers=0,
    )
    schemas = get_CMU_1_SMALL_REGION_schemas()
    expected_dims = ((2220, 2967), (387, 463), (1280, 431))
    expected_downsamples = (1.0,)
    for image_uri in tmp_path.glob("*"):
        img_idx = int(image_uri.name)
        t = TileDBOpenSlide.from_group_uri(str(image_uri))
        assert t.level_count == 1
        assert t.dimensions == expected_dims[img_idx]
        assert t.level_downsamples == expected_downsamples
        assert t.level_info[0] == LevelInfo(
            uri="", dimensions=schemas[img_idx].shape[:-3:-1]
        )
        region = t.read_region(level=0, location=(100, 100), size=(100, 200))
        assert isinstance(region, np.ndarray)
        assert region.ndim == 3
        assert region.dtype == np.uint8
        img = PIL.Image.fromarray(region)
        assert img.size == (100, 200)


def test_tiledb_to_ome_zarr_rountrip(tmp_path):
    # Take one image from CMU-1-Small-Region.ngff.zarr
    input_zarr_path = Path(get_path("CMU-1-Small-Region.ngff.zarr")) / "0.zarr"
    tiledb_path = tmp_path / "to_tiledb"
    output_zarr_path = tmp_path / "from_tiledb"

    # Store it to Tiledb
    OMEZarrConverter().to_tiledb(input_zarr_path, str(tiledb_path))
    # Store it back to NGFF Zarr
    OMEZarrConverter().from_tiledb(str(tiledb_path), output_zarr_path)

    # Same number of layers
    input_zarr_group = zarr.open_group(input_zarr_path, mode="r")
    tiledb_group = tiledb.Group(str(tiledb_path), mode="r")
    output_zarr_group = zarr.open_group(output_zarr_path, mode="r")
    assert len(input_zarr_group) == len(tiledb_group)
    assert len(input_zarr_group) == len(output_zarr_group)

    # Compare the .zattrs and .zgroup files
    for filename in ".zattrs", ".zgroup":
        with open(input_zarr_path / filename) as f:
            input_attrs = json.load(f)
        with open(output_zarr_path / filename) as f:
            output_attrs = json.load(f)
        assert input_attrs == output_attrs
