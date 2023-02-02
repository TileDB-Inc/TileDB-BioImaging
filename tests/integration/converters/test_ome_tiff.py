import json

import numpy as np
import PIL.Image
import pytest
import tifffile

import tiledb
from tests import get_path, get_schema
from tiledb.bioimg.converters import DATASET_TYPE, FMT_VERSION
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
from tiledb.bioimg.openslide import TileDBOpenSlide


@pytest.mark.parametrize("open_fileobj", [False, True])
@pytest.mark.parametrize("preserve_axes", [False, True])
def test_ome_tiff_converter(tmp_path, open_fileobj, preserve_axes):
    input_path = str(get_path("CMU-1-Small-Region.ome.tiff"))
    output_path = str(tmp_path)
    if open_fileobj:
        with open(input_path, "rb") as f:
            OMETiffConverter.to_tiledb(f, output_path, preserve_axes=preserve_axes)
    else:
        OMETiffConverter.to_tiledb(input_path, output_path, preserve_axes=preserve_axes)

    with TileDBOpenSlide(output_path) as t:
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

        for level in range(t.level_count):
            region_data = t.read_region((0, 0), level, t.level_dimensions[level])
            level_data = t.read_level(level)
            np.testing.assert_array_equal(region_data, level_data)


def test_ome_tiff_converter_different_dtypes(tmp_path):
    path = get_path("rand_uint16.ome.tiff")
    OMETiffConverter.to_tiledb(path, str(tmp_path))

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


@pytest.mark.parametrize("preserve_axes", [False, True])
def test_tiledb_to_ome_tiff_group_metadata(tmp_path, preserve_axes):
    input_path = get_path("CMU-1-Small-Region.ome.tiff")
    tiledb_path = tmp_path / "to_tiledb"
    OMETiffConverter.to_tiledb(
        input_path, str(tiledb_path), preserve_axes=preserve_axes
    )

    with TileDBOpenSlide(str(tiledb_path)) as t:
        group_properties = t.properties
        assert group_properties["dataset_type"] == DATASET_TYPE
        assert group_properties["fmt_version"] == FMT_VERSION
        assert isinstance(group_properties.get("pkg_version"), str)
        assert group_properties["axes"] == "CYX"
        assert group_properties["channels"] == json.dumps(
            ["Channel 0", "Channel 1", "Channel 2"]
        )

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


@pytest.mark.parametrize(
    "filename,num_series", [("CMU-1-Small-Region.ome.tiff", 3), ("UTM2GTIF.tiff", 1)]
)
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked,max_workers", [(False, 0), (True, 0), (True, 4)])
def test_tiledb_to_ome_tiff_roundtrip(
    tmp_path, filename, num_series, preserve_axes, chunked, max_workers
):
    input_path = get_path(filename)
    tiledb_path = tmp_path / "to_tiledb"
    output_path = tmp_path / "from_tiledb"
    to_tiledb_kwargs = dict(
        input_path=input_path,
        output_path=str(tiledb_path),
        preserve_axes=preserve_axes,
        chunked=chunked,
        max_workers=max_workers,
        reader_kwargs=dict(
            extra_tags=(
                "ModelPixelScaleTag",
                "ModelTiepointTag",
                "GeoKeyDirectoryTag",
                "GeoAsciiParamsTag",
            )
        ),
    )

    # Store it to Tiledb
    OMETiffConverter.to_tiledb(**to_tiledb_kwargs)
    # Store it back to NGFF Zarr
    OMETiffConverter.from_tiledb(str(tiledb_path), output_path)

    with tifffile.TiffFile(input_path) as t1, tifffile.TiffFile(output_path) as t2:
        compare_tifffiles(t1, t2)
        # only the first series is copied
        assert len(t1.series) == num_series
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

    OMETiffConverter.to_tiledb(input_path, str(tiledb_path), tiles=tiles)

    with TileDBOpenSlide(str(tiledb_path)) as t:
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
    compare_tiff_page_tags(
        s1.keyframe,
        s2.keyframe,
        skip={"PlanarConfiguration", "SampleFormat", "NewSubfileType"},
        ignore_value={
            "ImageDescription",
            "StripOffsets",
            "TileOffsets",
            "TileByteCounts",
            "SubIFDs",
            "XResolution",
            "YResolution",
        },
    )

    assert len(s1.pages) == len(s2.pages)
    assert len(s1.levels) == len(s2.levels)
    assert s1.levels[0] is s1
    assert s2.levels[0] is s2
    for l1, l2 in zip(s1.levels[1:], s2.levels[1:]):
        compare_tiff_page_series(l1, l2)


def compare_tiff_page_tags(p1, p2, skip=(), ignore_value=()):
    tags1 = [t for t in p1.tags.values() if t.name not in skip]
    tags2 = [t for t in p2.tags.values() if t.name not in skip]
    assert len(tags1) == len(tags2)
    for tag1, tag2 in zip(tags1, tags2):
        assert tag1.code == tag2.code
        assert tag1.name == tag2.name
        if tag1.name not in ignore_value:
            assert tag1.value == tag2.value, tag1.name
