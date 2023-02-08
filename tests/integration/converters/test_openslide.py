import json

import numpy as np
import openslide
import PIL.Image
import pytest

import tiledb
from tests import assert_image_similarity, get_path, get_schema
from tiledb.bioimg.converters import DATASET_TYPE, FMT_VERSION
from tiledb.bioimg.converters.openslide import OpenSlideConverter
from tiledb.bioimg.helpers import open_bioimg
from tiledb.bioimg.openslide import TileDBOpenSlide
from tiledb.cc import WebpInputFormat


@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked,max_workers", [(False, 0), (True, 0), (True, 4)])
@pytest.mark.parametrize(
    "compressor",
    [
        tiledb.ZstdFilter(level=0),
        tiledb.WebpFilter(WebpInputFormat.WEBP_RGBA, lossless=False),
        tiledb.WebpFilter(WebpInputFormat.WEBP_RGBA, lossless=True),
        tiledb.WebpFilter(WebpInputFormat.WEBP_NONE, lossless=True),
    ],
)
def test_openslide_converter(tmp_path, preserve_axes, chunked, max_workers, compressor):
    input_path = get_path("CMU-1-Small-Region.svs")
    output_path = str(tmp_path)
    OpenSlideConverter.to_tiledb(
        input_path,
        output_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
        max_workers=max_workers,
        compressor=compressor,
    )
    assert len(tiledb.Group(output_path)) == 1
    with open_bioimg(str(tmp_path / "l_0.tdb")) as A:
        if not preserve_axes:
            assert A.schema == get_schema(2220, 2967, 4, compressor=compressor)

    o = openslide.open_slide(input_path)
    with TileDBOpenSlide(output_path) as t:
        group_properties = t.properties
        assert group_properties["dataset_type"] == DATASET_TYPE
        assert group_properties["fmt_version"] == FMT_VERSION
        assert isinstance(group_properties.get("pkg_version"), str)
        assert group_properties["axes"] == "YXC"
        assert group_properties["channels"] == json.dumps(
            ["RED", "GREEN", "BLUE", "ALPHA"]
        )
        levels_group_meta = json.loads(group_properties["levels"])
        assert t.level_count == len(levels_group_meta)

        assert t.level_count == o.level_count
        assert t.dimensions == o.dimensions
        assert t.level_dimensions == o.level_dimensions
        assert t.level_downsamples == o.level_downsamples

        region_kwargs = dict(level=0, location=(123, 234), size=(1328, 1896))

        region = t.read_region(**region_kwargs)
        assert isinstance(region, np.ndarray)
        assert region.ndim == 3
        assert region.dtype == np.uint8
        t_img = PIL.Image.fromarray(region)
        o_img = o.read_region(**region_kwargs)
        if isinstance(compressor, tiledb.WebpFilter) and not compressor.lossless:
            assert_image_similarity(np.asarray(t_img), np.asarray(o_img))
        else:
            assert t_img == o_img

        for level in range(t.level_count):
            region_data = t.read_region((0, 0), level, t.level_dimensions[level])
            level_data = t.read_level(level)
            np.testing.assert_array_equal(region_data, level_data)


@pytest.mark.parametrize("preserve_axes", [False, True])
def test_openslide_converter_group_metadata(tmp_path, preserve_axes):
    input_path = get_path("CMU-1-Small-Region.svs")
    output_path = str(tmp_path)
    OpenSlideConverter.to_tiledb(input_path, output_path, preserve_axes=preserve_axes)

    with TileDBOpenSlide(output_path) as t:
        group_properties = t.properties
        assert group_properties["dataset_type"] == DATASET_TYPE
        assert group_properties["fmt_version"] == FMT_VERSION
        assert isinstance(group_properties.get("pkg_version"), str)
        assert group_properties["axes"] == "YXC"
        assert group_properties["channels"] == json.dumps(
            ["RED", "GREEN", "BLUE", "ALPHA"]
        )

        levels_group_meta = json.loads(group_properties["levels"])
        assert t.level_count == len(levels_group_meta)
        for level, level_meta in enumerate(levels_group_meta):
            assert level_meta["level"] == level
            assert level_meta["name"] == f"l_{level}.tdb"

            level_axes = level_meta["axes"]
            shape = level_meta["shape"]
            level_width, level_height = t.level_dimensions[level]
            assert level_axes == "YXC" if preserve_axes else "CYX"
            assert len(shape) == len(level_axes)
            assert shape[level_axes.index("C")] == 4
            assert shape[level_axes.index("X")] == level_width
            assert shape[level_axes.index("Y")] == level_height
