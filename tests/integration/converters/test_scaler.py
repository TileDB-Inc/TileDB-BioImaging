import pytest
from skimage.metrics import structural_similarity

from tests import get_path
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
from tiledb.bioimg.openslide import TileDBOpenSlide


@pytest.mark.parametrize("scale_factors", [[2, 4.0, 8, 16], [3.1, 11, 13]])
@pytest.mark.parametrize("chunked,max_workers", [(False, 0), (True, 0), (True, 4)])
@pytest.mark.parametrize("progressive", [True, False])
def test_scaler(tmp_path, scale_factors, chunked, max_workers, progressive):
    input_path = str(get_path("CMU-1-Small-Region.ome.tiff"))
    ground_path = str(tmp_path / "ground")
    test_path = str(tmp_path / "test")

    with open(input_path, "rb") as f:
        OMETiffConverter.to_tiledb(
            f,
            ground_path,
            pyramid_kwargs={
                "scale_factors": scale_factors,
                "scale_axes": "XY",
            },
        )

    with open(input_path, "rb") as f:
        OMETiffConverter.to_tiledb(
            f,
            test_path,
            pyramid_kwargs={
                "scale_factors": scale_factors,
                "scale_axes": "XY",
                "chunked": chunked,
                "progressive": progressive,
                "order": 1,
                "max_workers": max_workers,
            },
        )

    with TileDBOpenSlide.from_group_uri(ground_path) as ground:
        with TileDBOpenSlide.from_group_uri(test_path) as test:
            assert ground.level_count == test.level_count

            for level in range(ground.level_count):
                assert ground.level_dimensions[level] == test.level_dimensions[level]

                region_kwargs = dict(
                    level=level, location=(0, 0), size=test.level_dimensions[level]
                )
                assert (
                    structural_similarity(
                        ground.read_region(**region_kwargs),
                        test.read_region(**region_kwargs),
                        win_size=3,
                    )
                    > 0.95
                )
