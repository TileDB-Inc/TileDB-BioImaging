import numpy as np
import pytest
from PIL import Image

from tiledbimg.converters.openslide import OpenSlideReader


@pytest.fixture
def mocker_osd(mocker):
    mocker.patch("openslide.OpenSlide")


class TestOpenSlideReader:
    def test_osd_level_count(self, mocker_osd):
        reader = OpenSlideReader("")
        reader._osd.level_count = 5
        assert reader.level_count == 5

    def test_level_image(self, mocker, mocker_osd):
        data_img = np.random.randint(
            low=0, high=256, size=128 * 128 * 3, dtype=np.uint8
        )
        data_img = data_img.reshape(128, 128, 3)
        img = Image.fromarray(data_img, "RGB")
        expected = np.moveaxis(np.asarray(img), 2, 0)
        reader = OpenSlideReader("")
        reader._osd.read_region.return_value = img
        np.testing.assert_array_almost_equal(reader.level_image(0), expected)
