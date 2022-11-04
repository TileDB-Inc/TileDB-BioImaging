import glob
import json
import os
from pathlib import Path

import numpy as np
import PIL.Image
import tiledb
import zarr

from tests import get_CMU_1_SMALL_REGION_schemas, get_path
from tiledbimg.converters.ome_zarr import OMEZarrConverter
from tiledbimg.openslide import LevelInfo, TileDBOpenSlide


def test_ome_zarr_converter(tmp_path):
    test_image = os.path.join(get_path("CMU-1-Small-Region.ngff.zarr"), "0.zarr")
    output_path = str(tmp_path / os.path.basename(test_image))
    OMEZarrConverter().to_tiledb(str(test_image), output_path)

    schema = get_CMU_1_SMALL_REGION_schemas()[0]
    assert len(tiledb.Group(output_path)) == 1
    with tiledb.open(os.path.join(output_path, f"l_{0}.tdb")) as A:
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
    images = list(Path(get_path("CMU-1-Small-Region.ngff.zarr")).glob("*"))
    OMEZarrConverter().convert_images(
        images,
        str(tmp_path),
        level_min=0,
        max_workers=0,
    )
    schemas = get_CMU_1_SMALL_REGION_schemas()
    expected_dims = ((2220, 2967), (387, 463), (1280, 431))
    expected_downsamples = (1.0,)
    for img_idx, image_uri in enumerate(glob.glob(f"{str(tmp_path)}/*")):
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
    tmp_path.joinpath("to_tiledb").mkdir()
    tmp_path.joinpath("from_tiledb").mkdir()
    input_zarr = os.path.join(get_path("CMU-1-Small-Region.ngff.zarr"), "0.zarr")
    tiledb_image = str(
        tmp_path.joinpath("to_tiledb").joinpath(f"{os.path.basename(input_zarr)}")
    )
    output_zarr = str(
        tmp_path.joinpath("from_tiledb").joinpath(f"{os.path.basename(input_zarr)}")
    )

    # Store it to Tiledb
    OMEZarrConverter().to_tiledb(str(input_zarr), tiledb_image)
    # Store it back to NGFF Zarr
    OMEZarrConverter().from_tiledb(tiledb_image, output_zarr)

    zarr_image = zarr.open_group(input_zarr, mode="r")
    tiledb_image = tiledb.Group(tiledb_image, mode="r")
    roundtrip_zarr_image = zarr.open_group(output_zarr, mode="r")

    # Same number of layers
    assert len(list(zarr_image)) == len(tiledb_image)
    assert len(list(zarr_image)) == len(list(roundtrip_zarr_image))

    # Compare the .zattrs and .zgroup files
    with open(os.path.join(input_zarr, ".zattrs")) as input_zarr_attrs:
        with open(os.path.join(output_zarr, ".zattrs")) as expected_zarr_attrs:
            input_attrs = json.load(input_zarr_attrs)
            expected_attrs = json.load(expected_zarr_attrs)
            assert input_attrs == expected_attrs

    with open(os.path.join(input_zarr, ".zgroup")) as input_zarr_attrs:
        with open(os.path.join(output_zarr, ".zgroup")) as expected_zarr_attrs:
            input_attrs = json.load(input_zarr_attrs)
            expected_attrs = json.load(expected_zarr_attrs)
            assert input_attrs == expected_attrs
