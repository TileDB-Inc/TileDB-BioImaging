import json

import numpy as np
import PIL.Image
import pytest
import tiledb
import zarr

from tests import get_path, get_schema
from tiledbimg.converters.ome_zarr import OMEZarrConverter
from tiledbimg.openslide import TileDBOpenSlide

schemas = (get_schema(2220, 2967), get_schema(387, 463), get_schema(1280, 431))


@pytest.mark.parametrize("series_idx", [0, 1, 2])
def test_ome_zarr_converter(tmp_path, series_idx):
    input_path = get_path("CMU-1-Small-Region.ome.zarr") / str(series_idx)
    OMEZarrConverter().to_tiledb(input_path, str(tmp_path))

    # check the first (highest) resolution layer only
    schema = schemas[series_idx]
    with tiledb.open(str(tmp_path / f"l_{0}.tdb")) as A:
        assert A.schema == schema

    t = TileDBOpenSlide.from_group_uri(str(tmp_path))
    assert t.dimensions == t.level_dimensions[0] == schema.shape[:-3:-1]

    region_location = (100, 100)
    region_size = (300, 400)
    region = t.read_region(level=0, location=region_location, size=region_size)
    assert isinstance(region, np.ndarray)
    assert region.ndim == 3
    assert region.dtype == np.uint8
    img = PIL.Image.fromarray(region)
    assert img.size == (
        min(t.dimensions[0] - region_location[0], region_size[0]),
        min(t.dimensions[1] - region_location[1], region_size[1]),
    )


@pytest.mark.parametrize("series_idx", [0, 1, 2])
def test_tiledb_to_ome_zarr_rountrip(tmp_path, series_idx):
    input_path = get_path("CMU-1-Small-Region.ome.zarr") / str(series_idx)
    tiledb_path = tmp_path / "to_tiledb"
    output_path = tmp_path / "from_tiledb"

    cnv = OMEZarrConverter()
    # Store it to Tiledb
    cnv.to_tiledb(input_path, str(tiledb_path))
    # Store it back to NGFF Zarr
    cnv.from_tiledb(str(tiledb_path), output_path)

    # Same number of levels
    input_group = zarr.open_group(input_path, mode="r")
    tiledb_group = tiledb.Group(str(tiledb_path), mode="r")
    output_group = zarr.open_group(output_path, mode="r")
    assert len(input_group) == len(tiledb_group)
    assert len(input_group) == len(output_group)

    # Compare the .zattrs files
    with open(input_path / ".zattrs") as f:
        input_attrs = json.load(f)
        # ome-zarr-py replaces empty name with "/"
        name = input_attrs["multiscales"][0]["name"]
        if not name:
            input_attrs["multiscales"][0]["name"] = "/"
    with open(output_path / ".zattrs") as f:
        output_attrs = json.load(f)
    assert input_attrs == output_attrs

    # Compare the level arrays
    for i in range(len(input_group)):
        # Compare the .zarray files
        with open(input_path / str(i) / ".zarray") as f:
            input_zarray = json.load(f)
        with open(output_path / str(i) / ".zarray") as f:
            output_zarray = json.load(f)
        assert input_zarray == output_zarray

        # Compare the actual data
        input_zarray = zarr.open(input_path / str(i))
        output_zarray = zarr.open(output_path / str(i))
        np.testing.assert_array_equal(input_zarray[:], output_zarray[:])
