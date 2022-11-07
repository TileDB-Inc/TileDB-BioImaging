import json
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
import tiledb
import zarr

from tests import get_CMU_1_SMALL_REGION_schemas, get_path
from tiledbimg.converters.ome_zarr import OMEZarrConverter
from tiledbimg.openslide import TileDBOpenSlide


@pytest.mark.parametrize("series_idx", [0, 1, 2])
def test_ome_zarr_converter(tmp_path, series_idx):
    input_path = Path(get_path("CMU-1-Small-Region.ome.zarr")) / str(series_idx)
    OMEZarrConverter().to_tiledb(input_path, str(tmp_path))

    schema = get_CMU_1_SMALL_REGION_schemas()[series_idx]

    # check the first (highest) resolution layer only
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


def test_ome_zarr_converter_images(tmp_path):
    OMEZarrConverter().convert_images(
        Path(get_path("CMU-1-Small-Region.ome.zarr")).glob("[012]"),
        tmp_path,
        level_min=0,
        max_workers=0,
    )
    schemas = get_CMU_1_SMALL_REGION_schemas()
    for image_uri in tmp_path.glob("*"):
        img_idx = int(image_uri.name)
        t = TileDBOpenSlide.from_group_uri(str(image_uri))
        assert t.dimensions == t.level_dimensions[0] == schemas[img_idx].shape[:-3:-1]

        region = t.read_region(level=0, location=(100, 100), size=(100, 200))
        assert isinstance(region, np.ndarray)
        assert region.ndim == 3
        assert region.dtype == np.uint8
        img = PIL.Image.fromarray(region)
        assert img.size == (100, 200)


@pytest.mark.parametrize("series_idx", [0, 1, 2])
def test_tiledb_to_ome_zarr_rountrip(tmp_path, series_idx):
    input_zarr_path = Path(get_path("CMU-1-Small-Region.ome.zarr")) / str(series_idx)
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

    # Compare the .zattrs files
    with open(input_zarr_path / ".zattrs") as f:
        input_attrs = json.load(f)
        # ome-zarr-py replaces empty name with "/"
        name = input_attrs["multiscales"][0]["name"]
        if not name:
            input_attrs["multiscales"][0]["name"] = "/"
    with open(output_zarr_path / ".zattrs") as f:
        output_attrs = json.load(f)
    assert input_attrs == output_attrs

    # Compare the .zarray files
    for i in range(len(input_zarr_group)):
        with open(input_zarr_path / str(i) / ".zarray") as f:
            input_zarray = json.load(f)
        with open(output_zarr_path / str(i) / ".zarray") as f:
            output_zarray = json.load(f)
        assert input_zarray == output_zarray
