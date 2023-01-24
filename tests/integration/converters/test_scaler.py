import pytest

from tests import get_path
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
from tiledb.bioimg.converters.scale import ScalerMode
from tiledb.bioimg.openslide import TileDBOpenSlide


@pytest.mark.parametrize("scale_factor", [[2, 4, 8, 16], [2, 3, 5, 8], [3, 11, 13]])
@pytest.mark.parametrize(
    "scale_mode",
    [
        ScalerMode.NON_PROGRESSIVE,
        ScalerMode.CHUNKED_NON_PROGRESSIVE,
        ScalerMode.PROGRESSIVE,
        ScalerMode.CHUNKED_PROGRESSIVE,
    ],
)
def test_scaler(tmp_path, scale_factor, scale_mode):
    input_path = str(get_path("CMU-1-Small-Region.ome.tiff"))
    ground_path = str(tmp_path / "ground")
    test_path = str(tmp_path / "test")
    with open(input_path, "rb") as f:
        OMETiffConverter.to_tiledb(
            f,
            ground_path,
            generate_pyramid=True,
            pyramid_scale=scale_factor,
            pyramid_mode=ScalerMode.NON_PROGRESSIVE,
        )

    with open(input_path, "rb") as f:
        OMETiffConverter.to_tiledb(
            f,
            test_path,
            generate_pyramid=True,
            pyramid_scale=scale_factor,
            pyramid_mode=scale_mode,
        )

    with TileDBOpenSlide.from_group_uri(ground_path) as ground:
        with TileDBOpenSlide.from_group_uri(test_path) as test:
            assert ground.level_count == test.level_count

            for level in range(ground.level_count):
                assert ground.level_dimensions[level] == test.level_dimensions[level]

                # ground_data = ground.read_region((0, 0), level, ground.level_dimensions[level])
                # test_data = test.read_region((0, 0), level, test.level_dimensions[level])

                # np.testing.assert_allclose(ground_data, test_data, atol=5)
