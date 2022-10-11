import os

import numpy as np
import pytest
import zarr

from tests import rgb_to_5d
from tiledbimg.converters.ome_zarr import OMEZarrReader


@pytest.fixture
def mocker_zarr(mocker):
    mocker.patch("zarr.open")


@pytest.fixture
def data():
    level_count = 3
    np.random.seed(0)
    # Random RGB image
    data = np.random.randint(255, size=(256, 256, 3), dtype=np.uint8)
    return data, [rgb_to_5d(data) for _ in range(level_count)]


class TestOMEZarrReader:
    def test_ome_zarr_level_count(self, tmp_path, mocker_zarr, data):
        zarr_path = os.path.join(tmp_path, "test.zarr")
        reader = OMEZarrReader(zarr_path)
        reader._zarray = zarr.array(data[1])
        assert reader.level_count == 3

    def test_ome_zarr_level_image(self, tmp_path, mocker_zarr, data):
        zarr_path = os.path.join(tmp_path, "test.zarr")
        reader = OMEZarrReader(zarr_path)
        expected = data[0]
        zarr_array = zarr.array(data[1])
        # zarr_array.shape (level_count, 3, 1, 256, 256) = (t, c, z, y, x)
        reader._zarray.__getitem__.return_value = zarr_array
        actual = reader.level_image(0)
        np.testing.assert_array_equal(actual, expected)
