import json

import numpy as np
import PIL.Image
import pytest
import zarr

import tiledb
from tests import assert_image_similarity, get_path, get_schema
from tiledb.bioimg.converters import DATASET_TYPE, FMT_VERSION
from tiledb.bioimg.converters.ome_zarr import OMEZarrConverter
from tiledb.bioimg.helpers import iter_color, open_bioimg
from tiledb.bioimg.openslide import TileDBOpenSlide
from tiledb.cc import WebpInputFormat

schemas = (get_schema(2220, 2967), get_schema(387, 463), get_schema(1280, 431))


@pytest.mark.parametrize("series_idx", [0, 1, 2])
@pytest.mark.parametrize("preserve_axes", [False, True])
def test_ome_zarr_converter_source_reader_exception(
    tmp_path, series_idx, preserve_axes
):
    tiff_path = get_path("CMU-1-Small-Region.ome.tiff")
    output_reader = tmp_path / "to_tiledb_reader"

    with pytest.raises(FileExistsError) as excinfo:
        OMEZarrConverter.to_tiledb(
            tiff_path, str(output_reader), preserve_axes=preserve_axes
        )
    assert "FileExistsError" in str(excinfo)


@pytest.mark.parametrize("series_idx", [0, 1, 2])
@pytest.mark.parametrize("preserve_axes", [False, True])
def test_ome_zarr_converter_reader_source_consistent_output(
    tmp_path, series_idx, preserve_axes
):
    input_path = get_path("CMU-1-Small-Region.ome.zarr") / str(series_idx)

    output_path = tmp_path / "to_tiledb_path"
    output_reader = tmp_path / "to_tiledb_reader"

    OMEZarrConverter.to_tiledb(
        input_path, str(output_path), preserve_axes=preserve_axes
    )
    OMEZarrConverter.to_tiledb(
        input_path, str(output_reader), preserve_axes=preserve_axes
    )

    # check the first (highest) resolution layer only
    schema = schemas[series_idx]
    with open_bioimg(str(output_path / "l_0.tdb")) as A:
        with open_bioimg(str(output_reader / "l_0.tdb")) as B:
            if not preserve_axes:
                assert schema == A.schema == B.schema
            else:
                assert A.schema == B.schema


@pytest.mark.parametrize("series_idx", [0, 1, 2])
@pytest.mark.parametrize("preserve_axes", [False, True])
def test_ome_zarr_converter(tmp_path, series_idx, preserve_axes):
    input_path = get_path("CMU-1-Small-Region.ome.zarr") / str(series_idx)
    OMEZarrConverter.to_tiledb(input_path, str(tmp_path), preserve_axes=preserve_axes)

    # check the first (highest) resolution layer only
    schema = schemas[series_idx]
    with open_bioimg(str(tmp_path / "l_0.tdb")) as A:
        if not preserve_axes:
            assert A.schema == schema

    with TileDBOpenSlide(str(tmp_path)) as t:
        assert t.dimensions == t.level_dimensions[0] == schema.shape[:-3:-1]

        region_location = (100, 100)
        region_size = (300, 400)
        region = t.read_region(level=0, location=region_location, size=region_size)
        assert isinstance(region, np.ndarray)
        assert region.ndim == 3
        assert region.dtype == np.uint8
        img = PIL.Image.fromarray(region)
        assert img.size == (
            min(t.dimensions[0] - region_location[0], region_size[0]),
            min(t.dimensions[1] - region_location[1], region_size[1]),
        )

        for level in range(t.level_count):
            region_data = t.read_region((0, 0), level, t.level_dimensions[level])
            level_data = t.read_level(level)
            np.testing.assert_array_equal(region_data, level_data)


@pytest.mark.parametrize("series_idx", [0, 1, 2])
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
def test_ome_zarr_converter_rountrip(
    tmp_path, series_idx, preserve_axes, chunked, max_workers, compressor
):
    input_path = get_path("CMU-1-Small-Region.ome.zarr") / str(series_idx)
    tiledb_path = tmp_path / "to_tiledb"
    output_path = tmp_path / "from_tiledb"
    OMEZarrConverter.to_tiledb(
        input_path,
        str(tiledb_path),
        preserve_axes=preserve_axes,
        chunked=chunked,
        max_workers=max_workers,
        compressor=compressor,
    )
    # Store it back to NGFF Zarr
    OMEZarrConverter.from_tiledb(str(tiledb_path), str(output_path))

    # Same number of levels
    input_group = zarr.open_group(input_path, mode="r")
    tiledb_group = tiledb.Group(str(tiledb_path), mode="r")
    output_group = zarr.open_group(output_path, mode="r")
    assert len(input_group) == len(tiledb_group)
    assert len(input_group) == len(output_group)

    # Compare the .zattrs files
    with open(input_path / ".zattrs") as f:
        input_attrs = json.load(f)
        # ome-zarr-py replaces empty name with "/"
        name = input_attrs["multiscales"][0]["name"]
        if not name:
            input_attrs["multiscales"][0]["name"] = "/"
    with open(output_path / ".zattrs") as f:
        output_attrs = json.load(f)
    assert input_attrs == output_attrs

    # Compare the level arrays
    for i in range(len(input_group)):
        # Compare the .zarray files
        with open(input_path / str(i) / ".zarray") as f:
            input_zarray = json.load(f)
        with open(output_path / str(i) / ".zarray") as f:
            output_zarray = json.load(f)
        assert input_zarray == output_zarray

        # Compare the actual data
        input_array = zarr.open(input_path / str(i))[:]
        output_array = zarr.open(output_path / str(i))[:]
        if isinstance(compressor, tiledb.WebpFilter) and not compressor.lossless:
            assert_image_similarity(
                input_array.squeeze(),
                output_array.squeeze(),
                channel_axis=0,
                min_threshold=0.87,
            )
        else:
            np.testing.assert_array_equal(input_array, output_array)


def test_ome_zarr_converter_incremental(tmp_path):
    input_path = get_path("CMU-1-Small-Region.ome.zarr/0")

    OMEZarrConverter.to_tiledb(input_path, str(tmp_path), level_min=1)
    with TileDBOpenSlide(str(tmp_path)) as t:
        assert len(tiledb.Group(str(tmp_path))) == t.level_count == 1

    OMEZarrConverter.to_tiledb(input_path, str(tmp_path), level_min=0)
    with TileDBOpenSlide(str(tmp_path)) as t:
        assert len(tiledb.Group(str(tmp_path))) == t.level_count == 2

    OMEZarrConverter.to_tiledb(input_path, str(tmp_path), level_min=0)
    with TileDBOpenSlide(str(tmp_path)) as t:
        assert len(tiledb.Group(str(tmp_path))) == t.level_count == 2


@pytest.mark.parametrize("series_idx", [0, 1, 2])
@pytest.mark.parametrize("preserve_axes", [False, True])
def test_ome_zarr_converter_group_meta(tmp_path, series_idx, preserve_axes):
    input_path = get_path("CMU-1-Small-Region.ome.zarr") / str(series_idx)
    OMEZarrConverter.to_tiledb(input_path, str(tmp_path), preserve_axes=preserve_axes)

    with TileDBOpenSlide(str(tmp_path)) as t:
        group_properties = t.properties
        assert group_properties["dataset_type"] == DATASET_TYPE
        assert group_properties["fmt_version"] == FMT_VERSION
        assert isinstance(group_properties.get("pkg_version"), str)
        assert group_properties["axes"] == "TCZYX"
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
            assert level_axes == "TCZYX" if preserve_axes else "CYX"
            assert len(shape) == len(level_axes)
            assert shape[level_axes.index("C")] == 3
            assert shape[level_axes.index("X")] == level_width
            assert shape[level_axes.index("Y")] == level_height
            if preserve_axes:
                assert shape[level_axes.index("T")] == 1
                assert shape[level_axes.index("Z")] == 1


@pytest.mark.parametrize("series_idx", [0, 1, 2])
@pytest.mark.parametrize("preserve_axes", [False, True])
def test_ome_zarr_converter_channel_meta(tmp_path, series_idx, preserve_axes):
    input_path = get_path("CMU-1-Small-Region.ome.zarr") / str(series_idx)
    OMEZarrConverter.to_tiledb(input_path, str(tmp_path), preserve_axes=preserve_axes)

    with TileDBOpenSlide(str(tmp_path)) as t:
        group_properties = t.properties
        assert "metadata" in group_properties

        image_meta = json.loads(group_properties["metadata"])
        color_generator = iter_color(np.dtype(np.uint8))

        assert len(image_meta["channels"]) == 1
        assert "intensity" in image_meta["channels"]
        assert len(image_meta["channels"]["intensity"]) == 3
        for channel in image_meta["channels"]["intensity"]:
            assert channel["color"] == next(color_generator)
            assert channel["min"] == 0
            assert channel["max"] == 255
