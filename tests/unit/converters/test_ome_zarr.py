import os

import numpy as np
import pytest
import zarr

from tiledbimg.converters.ome_zarr import OMEZarrReader


@pytest.fixture
def mocker_tiff(mocker):
    mocker.patch("zarr.open")


class TestOMEZarrReader:
    def test_ome_zarr_level_count(self, tmp_path):
        zarr_path = os.path.join(tmp_path, "test.zarr")
        reader = OMEZarrReader(zarr_path)
        reader._zarray = zarr.array([1, 2, 3, 4])
        assert reader.level_count == 4
        reader._zarray = zarr.array([1, 2, 3, 4, 5])
        assert reader.level_count == 5

    def test_ome_zarr_level_downsamples(self, tmp_path):
        zarr_path = os.path.join(tmp_path, "test.zarr")
        reader = OMEZarrReader(zarr_path)
        assert reader.level_downsamples == ()

    def test_ome_zarr_level_image(self, tmp_path, mocker_tiff):
        zarr_path = os.path.join(tmp_path, "test.zarr")
        reader = OMEZarrReader(zarr_path)
        zarr_array = np.array(
            [[[[[[[0, 1], [2, 3]], [[4, 5], [6, 7, 8]]]]]]], dtype=object
        )
        reader._zarray.__getitem__.return_value = zarr_array
        actual = reader.level_image(0)
        expected = np.array(
            [[[[0, 1]], [[4, 5]]], [[[2, 3]], [[6, 7, 8]]]], dtype=object
        )
        np.testing.assert_array_equal(actual, expected)
