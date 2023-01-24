# import os
#
# os.add_dll_directory("C:/openslide/bin")
import json

import numpy as np
import openslide
import PIL.Image
import pytest

import tiledb
from tests import get_path, get_schema
from tiledb.bioimg.converters import DATASET_TYPE, FMT_VERSION
from tiledb.bioimg.converters.openslide import OpenSlideConverter
from tiledb.bioimg.openslide import TileDBOpenSlide


@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked,max_workers", [(False, 0), (True, 0), (True, 4)])
def test_openslide_converter(tmp_path, preserve_axes, chunked, max_workers):
    input_path = get_path("CMU-1-Small-Region.svs")
    output_path = str(tmp_path)
    to_tiledb_kwargs = dict(
        input_path=input_path,
        output_path=output_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
        max_workers=max_workers,
    )

    OpenSlideConverter.to_tiledb(**to_tiledb_kwargs)
    assert len(tiledb.Group(output_path)) == 1
    with tiledb.open(str(tmp_path / "l_0.tdb")) as A:
        if not preserve_axes:
            assert A.schema == get_schema(2220, 2967, 4)

    o = openslide.open_slide(input_path)
    tiledb_group = tiledb.Group(str(output_path), mode="r")

    with TileDBOpenSlide(output_path) as t:

        # Test group metadata
        levels_group_meta = json.loads(tiledb_group.meta["levels"])
        assert t.level_count == len(levels_group_meta)
        assert t.level_downsamples == tuple(
            [level["downsample_factor"] for level in levels_group_meta]
        )
        assert tiledb_group.meta["fmt_version"] == FMT_VERSION
        assert tiledb_group.meta["dataset_type"] == DATASET_TYPE

        assert t.level_count == o.level_count
        assert t.dimensions == o.dimensions
        assert t.level_dimensions == o.level_dimensions
        assert t.level_downsamples == o.level_downsamples

        region_kwargs = dict(level=0, location=(123, 234), size=(1328, 1896))

        region = t.read_region(**region_kwargs)
        assert isinstance(region, np.ndarray)
        assert region.ndim == 3
        assert region.dtype == np.uint8
        img = PIL.Image.fromarray(region)
        assert img == o.read_region(**region_kwargs)

        for level in range(t.level_count):
            region_data = t.read_region((0, 0), level, t.level_dimensions[level])
            level_data = t.read_level(level)
            np.testing.assert_array_equal(region_data, level_data)
