import numpy as np
import openslide
import PIL.Image
import pytest

import tiledb
from tests import get_path, get_schema
from tiledb.bioimg.converters.openslide import OpenSlideConverter
from tiledb.bioimg.openslide import TileDBOpenSlide


@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_openslide_converter(tmp_path, preserve_axes, chunked):
    input_path = get_path("CMU-1-Small-Region.svs")
    output_path = str(tmp_path)
    to_tiledb_kwargs = dict(
        input_path=input_path,
        output_path=output_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
    )

    OpenSlideConverter.to_tiledb(**to_tiledb_kwargs)
    assert len(tiledb.Group(output_path)) == 1
    with tiledb.open(str(tmp_path / "l_0.tdb")) as A:
        if not preserve_axes:
            assert A.schema == get_schema(2220, 2967)

    o = openslide.open_slide(input_path)
    with TileDBOpenSlide.from_group_uri(output_path) as t:

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
        assert img == o.read_region(**region_kwargs).convert("RGB")
