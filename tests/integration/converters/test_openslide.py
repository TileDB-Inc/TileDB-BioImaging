import numpy as np
import openslide
import PIL.Image
import tiledb

from tests import get_CMU_1_SMALL_REGION_schemas, get_path
from tiledbimg.converters.openslide import OpenSlideConverter
from tiledbimg.openslide import TileDBOpenSlide


def test_openslide_converter(tmp_path):
    svs_path = get_path("CMU-1-Small-Region.svs")
    OpenSlideConverter().to_tiledb(svs_path, str(tmp_path))

    schema = get_CMU_1_SMALL_REGION_schemas()[0]
    assert len(tiledb.Group(str(tmp_path))) == 1
    with tiledb.open(str(tmp_path / "l_0.tdb")) as A:
        assert A.schema == schema

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
