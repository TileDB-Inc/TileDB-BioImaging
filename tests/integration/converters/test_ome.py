import os

import pytest
import tiledb

from tests import get_path
from tiledbimg.converters.ome_tiff import OMETiffConverter
from tiledbimg.converters.ome_zarr import OMEZarrConverter
from tiledbimg.openslide import LevelInfo, TileDBOpenSlide

from . import CMU_1_SMALL_REGION


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
    dset = CMU_1_SMALL_REGION()
    schemas = dset.schema()

    assert (
        t.level_info[0] == LevelInfo(uri="", level=0, dimensions=schemas[0].shape)
        and t.level_info[1] == LevelInfo(uri="", level=1, dimensions=schemas[1].shape)
        and t.level_info[2] == LevelInfo(uri="", level=2, dimensions=schemas[2].shape)
    )
    assert t.level_count == 3
    assert t.dimensions == (2220, 2967)
    assert t.level_dimensions == ((2220, 2967), (387, 463), (1280, 431))
    assert t.level_downsamples == ()


def test_ome_tiff_converter(tmp_path):
    expected = CMU_1_SMALL_REGION().schema()
    src = get_path("CMU-1-Small-Region.ome.tiff")
    dest = os.path.join(tmp_path, ".tiledb")
    if not os.path.exists(dest):
        OMETiffConverter().convert_image(src, dest, level_min=0)
    with tiledb.open(os.path.join(dest, "l_0.tdb")) as A:
        assert A.schema == expected[0]
    with tiledb.open(os.path.join(dest, "l_1.tdb")) as A:
        assert A.schema == expected[1]
    with tiledb.open(os.path.join(dest, "l_2.tdb")) as A:
        assert A.schema == expected[2]


def test_ome_zarr_converter(tmp_path):
    expected = CMU_1_SMALL_REGION().schema()
    src = get_path("CMU-1-Small-Region.ome.zarr")
    dest = os.path.join(tmp_path, ".tiledb")
    if not os.path.exists(dest):
        OMEZarrConverter().convert_image(src, dest, level_min=0)
    with tiledb.open(os.path.join(dest, "l_0.tdb")) as A:
        assert A.schema == expected[0]
    with tiledb.open(os.path.join(dest, "l_1.tdb")) as A:
        assert A.schema == expected[1]
    with tiledb.open(os.path.join(dest, "l_2.tdb")) as A:
        assert A.schema == expected[2]
