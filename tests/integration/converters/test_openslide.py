import numpy as np
import openslide
import PIL.Image

import tiledb
from tests import get_path, get_schema
from tiledb.bioimg.converters.openslide import OpenSlideReader
from tiledb.bioimg.openslide import TileDBOpenSlide


def test_openslide_converter(tmp_path):
    svs_path = get_path("CMU-1-Small-Region.svs")
    OpenSlideReader.to_tiledb(svs_path, str(tmp_path))

    assert len(tiledb.Group(str(tmp_path))) == 1
    with tiledb.open(str(tmp_path / "l_0.tdb")) as A:
        assert A.schema == get_schema(2220, 2967)

    o = openslide.open_slide(svs_path)
    t = TileDBOpenSlide.from_group_uri(str(tmp_path))

    assert t.level_count == o.level_count
    assert t.dimensions == o.dimensions
    assert t.level_dimensions == o.level_dimensions
    assert t.level_downsamples == o.level_downsamples

    region = t.read_region(level=0, location=(100, 100), size=(300, 400))
    assert isinstance(region, np.ndarray)
    assert region.ndim == 3
    assert region.dtype == np.uint8
    img = PIL.Image.fromarray(region)
    assert img.size == (300, 400)
