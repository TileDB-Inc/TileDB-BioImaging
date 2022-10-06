from __future__ import annotations

import os

import openslide as osld
import pytest
import tiledb

from tests import download_from_s3
from tests.integration.converters import CMU_1_SMALL_REGION
from tiledbimg.open_slide import LevelInfo, SlideInfo, TileDBOpenSlide

g_uri = "s3://tiledb-isaiah2/jjdemo/test4-convert/C3N-02572-22.tdg"
svs_uri = "s3://tiledb-isaiah2/jjdemo/test4-convert/C3N-02572-22.svs"


def _check_level_info(num, info):
    assert info.level == num
    assert tiledb.object_type(info.uri) == "array"
    assert isinstance(info.schema, tiledb.ArraySchema)


class TestLevelInfo:
    def test_repr(self):
        test_data = CMU_1_SMALL_REGION()
        expected_level = 1
        expected_shape = (2220, 2967)
        li = LevelInfo(abspath="", level=expected_level, schema=test_data.schema()[0])
        assert str(li) == f"LevelInfo(level={expected_level}, shape={expected_shape})"

    def test_eq_operator(self, mocker):
        test_data = CMU_1_SMALL_REGION()
        test_level = 1
        linfo1 = LevelInfo(abspath="", level=test_level, schema=test_data.schema()[0])
        linfo2 = LevelInfo(abspath="", level=test_level, schema=test_data.schema()[1])
        assert linfo1 != linfo2
        linfo2 = LevelInfo(abspath="", level=test_level, schema=test_data.schema()[0])
        assert linfo1 == linfo2
        linfo2 = mocker.Mock()
        with pytest.raises(TypeError):
            assert linfo1 == linfo2

    def test_parse_level(self):
        # todo transform with regex or sth valid and generic
        test_level = 1
        expected_filename = f"test/path/root/levelfile_{test_level}.x"
        test_data = CMU_1_SMALL_REGION()
        linfo1 = LevelInfo(abspath="", level=test_level, schema=test_data.schema()[0])
        assert linfo1.parse_level(expected_filename) == 1
        test_level = 2
        expected_filename = f"test/path/root/levelfile_{test_level}.x"
        assert linfo1.parse_level(expected_filename) == 2
        expected_filename = "test/path/root/levelfile.x"
        with pytest.raises(ValueError):
            assert linfo1.parse_level(expected_filename)

    def test_from_array(self, tmp_path):
        test_data = CMU_1_SMALL_REGION()
        _ = tiledb.Array.create(
            os.path.join(tmp_path, "test.tdb"), test_data.schema()[0]
        )
        l0_uri = os.path.join(tmp_path, "test.tdb")
        l0_info = LevelInfo.from_array(l0_uri, 0)
        _check_level_info(0, l0_info)


class TestOpenSlide:
    def test_open_slide(self):
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


class TestSlideInfo:
    def test_slide_info(self):
        factor = 32
        slinfo = SlideInfo.from_group_uri(factor, g_uri)
        assert slinfo.factor == factor
        assert slinfo.slide == TileDBOpenSlide.from_group_uri(g_uri)
