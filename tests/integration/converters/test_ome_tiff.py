import numpy as np
import PIL.Image
import pytest
import tifffile
import tiledb
from tests import get_path, get_schema
from tiledb.bioimg.compressor_factory import ZstdArguments
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
from tiledb.bioimg.openslide import TileDBOpenSlide


@pytest.mark.parametrize("open_fileobj", [False, True])
@pytest.mark.parametrize("preserve_axes", [False, True])
def test_ome_tiff_converter(tmp_path, open_fileobj, preserve_axes):
    input_path = str(get_path("CMU-1-Small-Region.ome.tiff"))
    output_path = str(tmp_path)
    if open_fileobj:
        with open(input_path, "rb") as f:
            OMETiffConverter.to_tiledb(f, output_path, preserve_axes=preserve_axes, compressor_arguments=ZstdArguments(level=0))
    else:
        OMETiffConverter.to_tiledb(input_path, output_path, preserve_axes=preserve_axes, compressor_arguments=ZstdArguments(level=0))

    with TileDBOpenSlide.from_group_uri(output_path) as t:
        assert len(tiledb.Group(output_path)) == t.level_count == 2

        schemas = (get_schema(2220, 2967), get_schema(574, 768))
        assert t.dimensions == schemas[0].shape[:-3:-1]
        for i in range(t.level_count):
            assert t.level_dimensions[i] == schemas[i].shape[:-3:-1]
            with tiledb.open(str(tmp_path / f"l_{i}.tdb")) as A:
                if not preserve_axes:
                    assert A.schema == schemas[i]

        region = t.read_region(level=0, location=(100, 100), size=(300, 400))
        assert isinstance(region, np.ndarray)
        assert region.ndim == 3
        assert region.dtype == np.uint8
        img = PIL.Image.fromarray(region)
        assert img.size == (300, 400)


def test_ome_tiff_converter_different_dtypes(tmp_path):
    path = get_path("rand_uint16.ome.tiff")
    OMETiffConverter.to_tiledb(
        path, str(tmp_path), compressor_arguments=ZstdArguments(level=0)
    )

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

    # Store it to Tiledb
    OMETiffConverter.to_tiledb(
        input_path, str(tiledb_path), compressor_arguments=ZstdArguments(level=0)
    )
    # Store it back to NGFF Zarr
    OMETiffConverter.from_tiledb(str(tiledb_path), output_path)

    with tifffile.TiffFile(input_path) as t1, tifffile.TiffFile(output_path) as t2:
        compare_tifffiles(t1, t2)
        # only the first series is copied
        assert len(t1.series) == 3
        assert len(t2.series) == 1
        compare_tiff_page_series(t1.series[0], t2.series[0])


@pytest.mark.parametrize(
    "filename,dims",
    [
        ("single-channel.ome.tif", "YX"),
        ("z-series.ome.tif", "ZYX"),
        ("multi-channel.ome.tif", "CYX"),
        ("time-series.ome.tif", "TYX"),
        ("multi-channel-z-series.ome.tif", "CZYX"),
        ("multi-channel-time-series.ome.tif", "TCYX"),
        ("4D-series.ome.tif", "TZYX"),
        ("multi-channel-4D-series.ome.tif", "TCZYX"),
    ],
)
@pytest.mark.parametrize("tiles", [{}, {"X": 128, "Y": 128, "Z": 2, "C": 1, "T": 3}])
def test_ome_tiff_converter_artificial_rountrip(tmp_path, filename, dims, tiles):
    input_path = get_path(f"artificial-ome-tiff/{filename}")
    tiledb_path = tmp_path / "to_tiledb"
    output_path = tmp_path / "from_tiledb"

    OMETiffConverter.to_tiledb(
        input_path,
        str(tiledb_path),
        tiles=tiles,
        compressor_arguments=ZstdArguments(level=0),
    )

    with TileDBOpenSlide.from_group_uri(str(tiledb_path)) as t:
        assert len(tiledb.Group(str(tiledb_path))) == t.level_count == 1

    with tiledb.open(str(tiledb_path / "l_0.tdb")) as A:
        assert "".join(dim.name for dim in A.domain) == dims
        assert A.dtype == np.int8
        assert A.dim("X").tile == tiles.get("X", 439)
        assert A.dim("Y").tile == tiles.get("Y", 167)
        if A.domain.has_dim("Z"):
            assert A.dim("Z").tile == tiles.get("Z", 1)
        if A.domain.has_dim("C"):
            assert A.dim("C").tile == tiles.get("C", 3)
        if A.domain.has_dim("T"):
            assert A.dim("T").tile == tiles.get("T", 1)

    OMETiffConverter.from_tiledb(str(tiledb_path), output_path)
    with tifffile.TiffFile(input_path) as t1, tifffile.TiffFile(output_path) as t2:
        compare_tifffiles(t1, t2)
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

    assert s1.keyframe.hash == s2.keyframe.hash
    assert len(s1.pages) == len(s2.pages)
    assert len(s1.levels) == len(s2.levels)
    assert s1.levels[0] is s1
    assert s2.levels[0] is s2
    for l1, l2 in zip(s1.levels[1:], s2.levels[1:]):
        compare_tiff_page_series(l1, l2)
