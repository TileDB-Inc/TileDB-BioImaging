import pytest

from tests import get_path
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
from tiledb.bioimg.openslide import TileDBOpenSlide


@pytest.mark.parametrize("scale_factor", [[2, 4.0, 8, 16], [2, 3, 5, 8], [3.1, 11, 13]])
@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.parametrize("progressive", [True, False])
def test_scaler(tmp_path, scale_factor, chunked, progressive):
    input_path = str(get_path("CMU-1-Small-Region.ome.tiff"))
    ground_path = str(tmp_path / "ground")
    test_path = str(tmp_path / "test")

    with open(input_path, "rb") as f:
        OMETiffConverter.to_tiledb(
            f,
            ground_path,
            pyramid_kwargs={
                "scale_factors": scale_factor,
                "scale_axes": "XY",
            },
        )

    with open(input_path, "rb") as f:
        OMETiffConverter.to_tiledb(
            f,
            test_path,
            pyramid_kwargs={
                "scale_factors": scale_factor,
                "scale_axes": "XY",
                "chunked": chunked,
                "progressive": progressive,
                "order": 1,
            },
        )

    with TileDBOpenSlide.from_group_uri(ground_path) as ground:
        with TileDBOpenSlide.from_group_uri(test_path) as test:
            assert ground.level_count == test.level_count

            for level in range(ground.level_count):
                assert ground.level_dimensions[level] == test.level_dimensions[level]
