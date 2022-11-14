import numpy as np
import PIL.Image
import tifffile

import tiledb
from tests import get_path, get_schema
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
from tiledb.bioimg.openslide import TileDBOpenSlide


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


def test_tiledb_to_ome_tiff_rountrip(tmp_path):
    input_path = get_path("CMU-1-Small-Region.ome.tiff")
    tiledb_path = tmp_path / "to_tiledb"
    output_path = tmp_path / "from_tiledb"

    cnv = OMETiffConverter()
    # Store it to Tiledb
    cnv.to_tiledb(input_path, str(tiledb_path))
    # Store it back to NGFF Zarr
    cnv.from_tiledb(str(tiledb_path), output_path)

    with tifffile.TiffFile(input_path) as t1, tifffile.TiffFile(output_path) as t2:
        compare_tifffiles(t1, t2)
        # only the first series is copied
        assert len(t1.series) == 3
        assert len(t2.series) == 1
        compare_tiff_page_series(t1.series[0], t2.series[0])


def compare_tifffiles(t1, t2):
    assert t1.byteorder == t2.byteorder
    assert t1.is_bigtiff == t2.is_bigtiff
    assert t1.is_imagej == t2.is_imagej
    assert t1.is_ome == t2.is_ome


def compare_tiff_page_series(s1, s2):
    assert isinstance(s1, tifffile.TiffPageSeries)
    assert isinstance(s2, tifffile.TiffPageSeries)

    assert s1.shape == s2.shape
    assert s1.dtype == s2.dtype
    np.testing.assert_array_equal(s1.asarray(), s2.asarray())

    # XXX: if the tile length and/or width of the input are not a multiple of 16,
    # we don't write tile to the output. In this case the hashes don't match
    if s1.keyframe.tile is None or all(t % 16 == 0 for t in s1.keyframe.tile):
        assert s1.keyframe.hash == s2.keyframe.hash
    else:
        assert s2.keyframe.tile is None

    assert len(s1.pages) == len(s2.pages)
    assert len(s1.levels) == len(s2.levels)
    assert s1.levels[0] is s1
    assert s2.levels[0] is s2
    for l1, l2 in zip(s1.levels[1:], s2.levels[1:]):
        compare_tiff_page_series(l1, l2)
