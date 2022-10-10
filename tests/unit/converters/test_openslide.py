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

        # After substitution
        reader._osd.level_count = 3
        assert reader.level_count == 3

    def test_osd_level_downsamples(self, mocker_osd):
        reader = OpenSlideReader("")
        reader._osd.level_downsamples = (5.6, 7.4, 6.7)
        assert reader.level_downsamples == (5.6, 7.4, 6.7)

        # After substitution
        reader._osd.level_downsamples = (5.6, 7.4, 6.8)
        assert reader.level_downsamples == (5.6, 7.4, 6.8)

    def test_level_image(self, mocker, mocker_osd):
        data_img = np.random.randint(
            low=0, high=256, size=128 * 128 * 3, dtype=np.uint8
        )
        data_img = data_img.reshape(128, 128, 3)
        img = Image.fromarray(data_img, "RGB")
        expected = np.asarray(img).swapaxes(0, 1)
        reader = OpenSlideReader("")
        reader._osd.read_region.return_value = img
        np.testing.assert_array_almost_equal(reader.level_image(0), expected)
