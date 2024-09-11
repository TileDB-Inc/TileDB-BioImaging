import json

import numpy as np
import pytest
from PIL import Image, ImageChops

import tiledb
from tests import assert_image_similarity, get_path, get_schema
from tiledb.bioimg.converters import DATASET_TYPE, FMT_VERSION
from tiledb.bioimg.converters.png import PNGConverter
from tiledb.bioimg.helpers import open_bioimg
from tiledb.bioimg.openslide import TileDBOpenSlide
from tiledb.cc import WebpInputFormat


def test_png_converter(tmp_path):
    input_path = str(get_path("A01_s1--cell_outlines.png"))
    output_path = str(tmp_path)

    PNGConverter.to_tiledb(input_path, output_path)

    with TileDBOpenSlide(output_path) as t:
        assert len(tiledb.Group(output_path)) == t.level_count == 1
        schemas = get_schema(1080, 1080)

        # Storing the images as 3-channel images in CYX format
        # the slicing below using negative indexes to extract
        # the last two elements in schema's shape.
        assert t.dimensions == schemas.shape[:-3:-1]
        for i in range(t.level_count):
            assert t.level_dimensions[i] == schemas.shape[:-3:-1]
            with open_bioimg(str(tmp_path / f"l_{i}.tdb")) as A:
                assert A.schema == schemas

        region = t.read_region(level=0, location=(100, 100), size=(300, 400))
        assert isinstance(region, np.ndarray)
        assert region.ndim == 3
        assert region.dtype == np.uint8
        img = Image.fromarray(region)
        assert img.size == (300, 400)

        for level in range(t.level_count):
            region_data = t.read_region((0, 0), level, t.level_dimensions[level])
            level_data = t.read_level(level)
            np.testing.assert_array_equal(region_data, level_data)


@pytest.mark.parametrize("filename", ["A01_s1--cell_outlines.png"])
def test_png_converter_group_metadata(tmp_path, filename):
    input_path = get_path(filename)
    tiledb_path = str(tmp_path / "to_tiledb")
    PNGConverter.to_tiledb(input_path, tiledb_path, preserve_axes=False)

    with TileDBOpenSlide(tiledb_path) as t:
        group_properties = t.properties
        assert group_properties["dataset_type"] == DATASET_TYPE
        assert group_properties["fmt_version"] == FMT_VERSION
        assert isinstance(group_properties["pkg_version"], str)
        assert group_properties["axes"] == "XYC"
        assert group_properties["channels"] == json.dumps(["RED", "GREEN", "BLUE"])

        levels_group_meta = json.loads(group_properties["levels"])
        assert t.level_count == len(levels_group_meta)
        for level, level_meta in enumerate(levels_group_meta):
            assert level_meta["level"] == level
            assert level_meta["name"] == f"l_{level}.tdb"

            level_axes = level_meta["axes"]
            shape = level_meta["shape"]
            level_width, level_height = t.level_dimensions[level]
            assert level_axes == "CYX"
            assert len(shape) == len(level_axes)
            assert shape[level_axes.index("C")] == 3
            assert shape[level_axes.index("X")] == level_width
            assert shape[level_axes.index("Y")] == level_height


def compare_png(p1: Image, p2: Image, lossless: bool = True):
    diff = ImageChops.difference(p1, p2)
    if lossless:
        assert diff.getbbox() is None
    else:
        assert_image_similarity(np.array(p1), np.array(p2), channel_axis=0)


@pytest.mark.parametrize("filename", ["A01_s1--cell_outlines.png"])
@pytest.mark.parametrize("preserve_axes", [False, True])
# PIL.Image does not support chunked reads/writes
@pytest.mark.parametrize("chunked", [False])
@pytest.mark.parametrize(
    "compressor, lossless",
    [
        (tiledb.ZstdFilter(level=0), True),
        (tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=False), False),
        (tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=True), True),
        (tiledb.WebpFilter(WebpInputFormat.WEBP_NONE, lossless=True), True),
    ],
)
def test_png_converter_roundtrip(
    tmp_path, filename, preserve_axes, chunked, compressor, lossless
):
    input_path = get_path(filename)
    tiledb_path = str(tmp_path / "to_tiledb")
    output_path = str(tmp_path / "from_tiledb")
    PNGConverter.to_tiledb(
        input_path,
        tiledb_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
        compressor=compressor,
        log=False,
    )
    # Store it back to PNG
    PNGConverter.from_tiledb(tiledb_path, output_path)
    compare_png(Image.open(input_path), Image.open(output_path), lossless=lossless)
