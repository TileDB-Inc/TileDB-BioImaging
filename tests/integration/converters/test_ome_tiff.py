import numpy as np
import PIL.Image
import tiledb

from tests import get_path, get_schema
from tiledbimg.converters.ome_tiff import OMETiffConverter
from tiledbimg.openslide import TileDBOpenSlide


def test_ome_tiff_converter(tmp_path):
    OMETiffConverter().to_tiledb(get_path("CMU-1-Small-Region.ome.tiff"), str(tmp_path))

    t = TileDBOpenSlide.from_group_uri(str(tmp_path))
    assert len(tiledb.Group(str(tmp_path))) == t.level_count == 2

    schemas = (get_schema(2220, 2967), get_schema(574, 768))
    assert t.dimensions == schemas[0].shape[:-3:-1]
    for i in range(t.level_count):
        assert t.level_dimensions[i] == schemas[i].shape[:-3:-1]
        with tiledb.open(str(tmp_path / f"l_{i}.tdb")) as A:
            assert A.schema == schemas[i]

    region = t.read_region(level=0, location=(100, 100), size=(300, 400))
    assert isinstance(region, np.ndarray)
    assert region.ndim == 3
    assert region.dtype == np.uint8
    img = PIL.Image.fromarray(region)
    assert img.size == (300, 400)


def test_ome_tiff_converter_different_dtypes(tmp_path):
    path = get_path("rand_uint16.ome.tiff")
    OMETiffConverter().to_tiledb(path, str(tmp_path))

    assert len(tiledb.Group(str(tmp_path))) == 3
    with tiledb.open(str(tmp_path / "l_0.tdb")) as A:
        assert A.schema.domain.dtype == np.uint32
        assert A.attr(0).dtype == np.uint16
    with tiledb.open(str(tmp_path / "l_1.tdb")) as A:
        assert A.schema.domain.dtype == np.uint16
        assert A.attr(0).dtype == np.uint16
    with tiledb.open(str(tmp_path / "l_2.tdb")) as A:
        assert A.schema.domain.dtype == np.uint16
        assert A.attr(0).dtype == np.uint16
