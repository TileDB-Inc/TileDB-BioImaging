import pytest

from tests import get_path
from tests.integration.converters import CMU_1_SMALL_REGION
from tiledbimg.open_slide import LevelInfo, TileDBOpenSlide


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
    reference_schema = dset.schema()

    assert (
        t.level_info[0] == LevelInfo(abspath="", level=0, schema=reference_schema[0])
        and t.level_info[1]
        == LevelInfo(abspath="", level=1, schema=reference_schema[1])
        and t.level_info[2]
        == LevelInfo(abspath="", level=2, schema=reference_schema[2])
    )
    assert t.level_count == 3
    assert t.dimensions == (2220, 2967)
    assert t.level_dimensions == ((2220, 2967), (387, 463), (1280, 431))
    assert t.level_downsamples is None
