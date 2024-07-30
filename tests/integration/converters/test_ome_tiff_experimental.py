import numpy as np
import pytest
import tifffile

import tiledb
from tests import assert_image_similarity, get_path
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
from tiledb.bioimg.helpers import open_bioimg
from tiledb.bioimg.openslide import TileDBOpenSlide
from tiledb.cc import WebpInputFormat


# We need to expand on the test files. Most of the test files we have currently are not memory
# contiguous and the ones that are not RGB files to test the different compressors
@pytest.mark.parametrize("filename,num_series", [("UTM2GTIF.tiff", 1)])
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked,max_workers", [(False, 0), (True, 0), (True, 4)])
@pytest.mark.parametrize(
    "compressor",
    [tiledb.ZstdFilter(level=0)],
)
def test_ome_tiff_converter_exclude_original_metadata(
    tmp_path, filename, num_series, preserve_axes, chunked, max_workers, compressor
):
    # The image is not RGB to use the WbeP compressor
    if isinstance(compressor, tiledb.WebpFilter) and filename == "UTM2GTIF.tiff":
        pytest.skip(f"WebPFilter cannot be applied to {filename}")

    input_path = get_path(filename)
    tiledb_path = tmp_path / "to_tiledb"
    exprimental_path = tmp_path / "experimental"
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

    OMETiffConverter.to_tiledb(
        input_path,
        str(exprimental_path),
        preserve_axes=preserve_axes,
        chunked=True,
        max_workers=max_workers,
        compressor=compressor,
        log=False,
        exclude_metadata=True,
        experimental_reader=True,
    )

    with TileDBOpenSlide(str(tiledb_path)) as t:
        with TileDBOpenSlide(str(exprimental_path)) as e:
            assert t.level_count == e.level_count

            for level in range(t.level_count):
                np.testing.assert_array_equal(t.read_level(level), e.read_level(level))


@pytest.mark.parametrize("filename,num_series", [("UTM2GTIF.tiff", 1)])
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked,max_workers", [(True, 0), (True, 4)])
@pytest.mark.parametrize(
    "compressor",
    [tiledb.ZstdFilter(level=0)],
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
        experimental_reader=True,
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
    experimental_path = tmp_path / "_experimental"
    output_path = tmp_path / "from_tiledb"

    OMETiffConverter.to_tiledb(input_path, str(tiledb_path), tiles=tiles)
    OMETiffConverter.to_tiledb(
        input_path,
        str(experimental_path),
        tiles=tiles,
        experimental_reader=True,
        chunked=True,
        max_workers=16,
    )

    with TileDBOpenSlide(str(experimental_path)) as t:
        assert len(tiledb.Group(str(experimental_path))) == t.level_count == 1

    with open_bioimg(str(experimental_path / "l_0.tdb")) as A:
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
