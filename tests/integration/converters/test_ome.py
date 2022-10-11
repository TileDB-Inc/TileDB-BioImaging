import os
import shutil

import pytest
import tiledb

from tests import get_CMU_1_SMALL_REGION_schemas, get_path
from tiledbimg.converters.ome_tiff import OMETiffConverter
from tiledbimg.converters.ome_zarr import OMEZarrConverter
from tiledbimg.openslide import LevelInfo, TileDBOpenSlide


@pytest.mark.parametrize(
    "format_path", ["CMU-1-Small-Region.tiledb", "CMU-1-Small-Region-Zarr.tiledb"]
)
def test_ome(format_path):
    # TODO: We need to find better test data.
    # This data has already been downsampled without preserving the original levels.

    # import openslide as osld
    # import tifffile
    # ometiff_img = tifffile.TiffFile(get_path("CMU-1-Small-Region.ome.tiff"))
    # os_img = osld.open_slide(get_path("CMU-1-Small-Region.svs.tiff"))

    t = TileDBOpenSlide.from_group_uri(get_path(format_path))
    schemas = get_CMU_1_SMALL_REGION_schemas()

    for i in range(0, 3):
        assert t.level_info[i] == LevelInfo(
            uri="", level=i, dimensions=schemas[i].shape
        )

    assert t.level_count == 3
    assert t.dimensions == (2220, 2967)
    assert t.level_dimensions == ((2220, 2967), (387, 463), (1280, 431))
    assert t.level_downsamples == ()


@pytest.mark.parametrize(
    "converter,path",
    [
        (OMEZarrConverter, "CMU-1-Small-Region.ome.zarr"),
        (OMETiffConverter, "CMU-1-Small-Region.ome.tiff"),
    ],
)
def test_ome_converter(tmp_path, converter, path):
    expected = get_CMU_1_SMALL_REGION_schemas()
    src = get_path(path)
    dest = tmp_path / "tiledb"
    dest.mkdir()
    converter().convert_image(src, dest.as_uri(), level_min=0)

    for i in range(0, 3):
        with tiledb.open(os.path.join(dest.as_uri(), f"l_{i}.tdb")) as A:
            assert A.schema == expected[i]
    group = tiledb.Group(dest.as_uri())
    actual = list(group)
    assert len(expected) == len(actual)


@pytest.mark.parametrize("max_workers", [0, 1, 2])
def test_ome_tiff_converter_group(tmp_path, max_workers):
    src = get_path("CMU-1-Small-Region.ome.tiff")

    # Replicate image file in other location
    src2 = tmp_path / "src2"
    src2.mkdir()
    shutil.copy(src, src2)
    shutil.move(
        os.path.join(src2, "CMU-1-Small-Region.ome.tiff"),
        os.path.join(src2, "CMU-2-Small-Region.ome.tiff"),
    )

    image_list = [src, os.path.join(src2, "CMU-2-Small-Region.ome.tiff")]
    groups = tmp_path / "groups"
    groups.mkdir()
    if not os.path.exists(groups.as_uri()):
        OMETiffConverter().convert_images(
            image_list, groups.as_uri(), level_min=0, max_workers=max_workers
        )

    grp_1 = tiledb.Group(os.path.join(groups.as_uri(), "CMU-1-Small-Region.ome"), "r")
    assert grp_1.meta["original_filename"] == src
    grp_2 = tiledb.Group(os.path.join(groups.as_uri(), "CMU-2-Small-Region.ome"), "r")
    assert grp_2.meta["original_filename"] == os.path.join(
        src2, "CMU-2-Small-Region.ome.tiff"
    )
