import itertools
import json

import numpy as np
import PIL.Image
import pytest
import tifffile

import tiledb
from tests import assert_image_similarity, get_path, get_schema
from tiledb.bioimg.converters import DATASET_TYPE, FMT_VERSION
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
from tiledb.bioimg.helpers import open_bioimg
from tiledb.bioimg.openslide import TileDBOpenSlide
from tiledb.cc import WebpInputFormat


def test_ome_tiff_converter(tmp_path):
    input_path = str(get_path("CMU-1-Small-Region.ome.tiff"))
    output_path = str(tmp_path)

    OMETiffConverter.to_tiledb(input_path, output_path)

    with TileDBOpenSlide(output_path) as t:
        assert len(tiledb.Group(output_path)) == t.level_count == 2

        schemas = (get_schema(2220, 2967), get_schema(574, 768))
        assert t.dimensions == schemas[0].shape[:-3:-1]
        for i in range(t.level_count):
            assert t.level_dimensions[i] == schemas[i].shape[:-3:-1]
            with open_bioimg(str(tmp_path / f"l_{i}.tdb")) as A:
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
    with open_bioimg(str(tmp_path / "l_0.tdb")) as A:
        assert A.schema.domain.dtype == np.uint32
        assert A.attr(0).dtype == np.uint16
    with open_bioimg(str(tmp_path / "l_1.tdb")) as A:
        assert A.schema.domain.dtype == np.uint16
        assert A.attr(0).dtype == np.uint16
    with open_bioimg(str(tmp_path / "l_2.tdb")) as A:
        assert A.schema.domain.dtype == np.uint16
        assert A.attr(0).dtype == np.uint16


@pytest.mark.parametrize(
    "filename", ["CMU-1-Small-Region.ome.tiff", "CMU-1-Small-Region-rgb.ome.tiff"]
)
def test_ome_tiff_converter_group_metadata(tmp_path, filename):
    input_path = get_path(filename)
    tiledb_path = tmp_path / "to_tiledb"
    OMETiffConverter.to_tiledb(input_path, str(tiledb_path))

    with TileDBOpenSlide(str(tiledb_path)) as t:
        group_properties = t.properties
        assert group_properties["dataset_type"] == DATASET_TYPE
        assert group_properties["fmt_version"] == FMT_VERSION
        assert isinstance(group_properties.get("pkg_version"), str)
        assert group_properties["axes"] == "CYX"
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


@pytest.mark.parametrize(
    "filename,num_series", [("CMU-1-Small-Region.ome.tiff", 3), ("UTM2GTIF.tiff", 1)]
)
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked,max_workers", [(False, 0), (True, 0), (True, 4)])
@pytest.mark.parametrize(
    "compressor",
    [
        tiledb.ZstdFilter(level=0),
        tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=False),
        tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=True),
        tiledb.WebpFilter(WebpInputFormat.WEBP_NONE, lossless=True),
    ],
)
def test_ome_tiff_converter_exclude_original_metadata(
    tmp_path, filename, num_series, preserve_axes, chunked, max_workers, compressor
):
    if isinstance(compressor, tiledb.WebpFilter) and filename == "UTM2GTIF.tiff":
        pytest.skip(f"WebPFilter cannot be applied to {filename}")

    input_path = get_path(filename)
    tiledb_path = tmp_path / "to_tiledb"
    OMETiffConverter.to_tiledb(
        input_path,
        str(tiledb_path),
        preserve_axes=preserve_axes,
        chunked=chunked,
        max_workers=max_workers,
        compressor=compressor,
        log=False,
        exclude_metadata=True,
    )

    with TileDBOpenSlide(str(tiledb_path)) as t:
        assert t.properties["original_metadata"] == "{}"


@pytest.mark.parametrize(
    "filename,num_series", [("CMU-1-Small-Region.ome.tiff", 3), ("UTM2GTIF.tiff", 1)]
)
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked,max_workers", [(False, 0), (True, 0), (True, 4)])
@pytest.mark.parametrize(
    "compressor",
    [
        tiledb.ZstdFilter(level=0),
        tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=False),
        tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=True),
        tiledb.WebpFilter(WebpInputFormat.WEBP_NONE, lossless=True),
    ],
)
def test_ome_tiff_converter_roundtrip(
    tmp_path, filename, num_series, preserve_axes, chunked, max_workers, compressor
):
    if isinstance(compressor, tiledb.WebpFilter) and filename == "UTM2GTIF.tiff":
        pytest.skip(f"WebPFilter cannot be applied to {filename}")

    input_path = get_path(filename)
    tiledb_path = tmp_path / "to_tiledb"
    output_path = tmp_path / "from_tiledb"
    OMETiffConverter.to_tiledb(
        input_path,
        str(tiledb_path),
        preserve_axes=preserve_axes,
        chunked=chunked,
        max_workers=max_workers,
        compressor=compressor,
        log=False,
        reader_kwargs=dict(
            extra_tags=(
                "ModelPixelScaleTag",
                "ModelTiepointTag",
                "GeoKeyDirectoryTag",
                "GeoAsciiParamsTag",
            )
        ),
    )
    # Store it back to NGFF Zarr
    OMETiffConverter.from_tiledb(str(tiledb_path), str(output_path))

    with tifffile.TiffFile(input_path) as t1, tifffile.TiffFile(output_path) as t2:
        compare_tiff(t1, t2, lossless=False)


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

    with open_bioimg(str(tiledb_path / "l_0.tdb")) as A:
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

    OMETiffConverter.from_tiledb(str(tiledb_path), str(output_path))
    with tifffile.TiffFile(input_path) as t1, tifffile.TiffFile(output_path) as t2:
        compare_tiff(t1, t2, lossless=True)


def compare_tiff(t1: tifffile.TiffFile, t2: tifffile.TiffFile, lossless: bool = True):
    assert len(t1.series[0].levels) == len(t2.series[0].levels)

    for l1, l2 in zip(t1.series[0].levels, t2.series[0].levels):
        assert l1.axes.replace("S", "C") == l2.axes.replace("S", "C")
        assert l1.shape == l2.shape
        assert l1.dtype == l2.dtype
        assert l1.nbytes == l2.nbytes

        if lossless:
            np.testing.assert_array_equal(l1.asarray(), l2.asarray())
        else:
            assert_image_similarity(l1.asarray(), l2.asarray(), channel_axis=0)


compressors = [
    None,
    tiledb.ZstdFilter(level=0),
    tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=False),
    tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=True),
]


@pytest.mark.parametrize(
    "filename", ["CMU-1-Small-Region.ome.tiff", "CMU-1-Small-Region-rgb.ome.tiff"]
)
@pytest.mark.parametrize(
    "compressor_0, compressor_1", itertools.product(compressors, compressors)
)
@pytest.mark.parametrize("mapping", [True, False])
def test_ome_tiff_converter_different_compressors(
    tmp_path, filename, compressor_0, compressor_1, mapping
):
    input_path = str(get_path(filename))
    output_path = str(tmp_path)
    filters = {0: compressor_0, 1: compressor_1}
    if not compressor_0 and not compressor_1:
        # Test the default filter ZstdFilter when compressor==None
        OMETiffConverter.to_tiledb(input_path, output_path, compressor=None)
        assert len(tiledb.Group(str(tmp_path))) == 2
        with open_bioimg(str(tmp_path / "l_0.tdb")) as A:
            assert len(A.schema.attr(0).filters) == 1
            assert isinstance(A.schema.attr(0).filters[0], tiledb.ZstdFilter)
        with open_bioimg(str(tmp_path / "l_1.tdb")) as A:
            assert len(A.schema.attr(0).filters) == 1
            assert isinstance(A.schema.attr(0).filters[0], tiledb.ZstdFilter)
    elif compressor_0 and compressor_1:
        # Both compressors are given
        OMETiffConverter.to_tiledb(input_path, output_path, compressor=filters)
        assert len(tiledb.Group(str(tmp_path))) == 2
        with open_bioimg(str(tmp_path / "l_0.tdb")) as A:
            assert len(A.schema.attr(0).filters) == 1
            assert isinstance(A.schema.attr(0).filters[0], type(compressor_0))
        with open_bioimg(str(tmp_path / "l_1.tdb")) as A:
            assert len(A.schema.attr(0).filters) == 1
            assert isinstance(A.schema.attr(0).filters[0], type(compressor_1))
    else:
        filtered_compressors = {k: v for k, v in filters.items() if v is not None}
        if mapping:
            # Error out if number of compressors in mapping not equal to levels
            with pytest.raises(ValueError) as ve:
                OMETiffConverter.to_tiledb(
                    input_path, output_path, compressor=filtered_compressors
                )
                assert (
                    "Compressor filter mapping does not map every level to a Filter"
                    in str(ve.value)
                )
        else:
            # Apart from mapping accept also a single filter
            single_compressor = next(iter(filtered_compressors.values()))
            OMETiffConverter.to_tiledb(
                input_path, output_path, compressor=single_compressor
            )
            with open_bioimg(str(tmp_path / "l_0.tdb")) as A:
                assert len(A.schema.attr(0).filters) == 1
                assert isinstance(A.schema.attr(0).filters[0], type(single_compressor))
            with open_bioimg(str(tmp_path / "l_1.tdb")) as A:
                assert len(A.schema.attr(0).filters) == 1
                assert isinstance(A.schema.attr(0).filters[0], type(single_compressor))
