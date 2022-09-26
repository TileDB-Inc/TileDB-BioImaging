import os

import openslide as osld
import tiledb

from tiledbimg.open_slide import LevelInfo, SlideInfo, TileDBOpenSlide

from . import download_from_s3

g_uri = "s3://tiledb-isaiah2/jjdemo/test4-convert/C3N-02572-22.tdg"
svs_uri = "s3://tiledb-isaiah2/jjdemo/test4-convert/C3N-02572-22.svs"


def _check_level_info(num, info):
    assert info.level == num
    assert tiledb.object_type(info.uri) == "array"
    assert isinstance(info.schema, tiledb.ArraySchema)


def test_level_info():
    l0_uri = os.path.join(g_uri, "l_0.tdb")
    l0_info = LevelInfo.from_array(l0_uri, 0)
    _check_level_info(0, l0_info)


def test_open_slide():
    t = TileDBOpenSlide.from_group_uri(g_uri)

    for l_num, l_info in enumerate(t._level_infos):
        _check_level_info(l_num, l_info)

    os_img = osld.open_slide(download_from_s3(svs_uri))
    assert t.dimensions == os_img.dimensions
    assert t.level_count == os_img.level_count
    assert t.level_dimensions == os_img.level_dimensions
    assert t.level_downsamples == os_img.level_downsamples
    assert (
        t.get_best_level_for_downsample(32)
        == 2
        == os_img.get_best_level_for_downsample(32)
    )
    assert (
        t.get_best_level_for_downsample(3.9)
        == 0
        == os_img.get_best_level_for_downsample(3.9)
    )
    assert (
        t.get_best_level_for_downsample(4.1)
        == 1
        == os_img.get_best_level_for_downsample(4.1)
    )
    assert (
        t.get_best_level_for_downsample(2.9)
        == 0
        == os_img.get_best_level_for_downsample(2.9)
    )
    assert (
        t.get_best_level_for_downsample(1)
        == 0
        == os_img.get_best_level_for_downsample(1)
    )


def test_slide_info():
    factor = 32
    slinfo = SlideInfo.from_group_uri(factor, g_uri)

    assert slinfo.factor == factor
    assert slinfo.slide == TileDBOpenSlide.from_group_uri(g_uri)
