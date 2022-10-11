import numpy as np
import pytest

from tiledbimg.converters.ome_tiff import OMETiffReader


@pytest.fixture
def mocker_tiff(mocker):
    mocker.patch("tifffile.TiffFile")


class TestOMETiffReader:
    def test_ome_tiff_level_count(self, mocker_tiff):
        reader = OMETiffReader("")
        reader._tiff_series = [1, 2, 3, 4]
        assert reader.level_count == 4
        reader._tiff_series = [1, 2, 3, 4, 5]
        assert reader.level_count == 5

    def test_ome_tiff_level_image(self, mocker_tiff):
        reader = OMETiffReader("")
        tiff_array = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        reader._tiff_series.__getitem__().asarray.return_value = tiff_array
        actual = reader.level_image(0)
        expected = np.array([[[0, 4], [2, 6]], [[1, 5], [3, 7]]])
        np.testing.assert_array_equal(actual, expected)
