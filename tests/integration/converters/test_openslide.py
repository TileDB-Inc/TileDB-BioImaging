import numpy as np
import openslide
import PIL.Image

from tests import get_path
from tiledbimg.converters.openslide import OpenSlideConverter
from tiledbimg.openslide import LevelInfo, TileDBOpenSlide


def test_openslide_converter(tmp_path):
    svs_path = get_path("s3://tiledb-isaiah2/jjdemo/test4-convert/C3N-02572-22.svs")

    os_img = openslide.open_slide(svs_path)
    assert os_img.level_count == 3
    assert os_img.dimensions == (19919, 21702)
    assert os_img.level_dimensions == ((19919, 21702), (4979, 5425), (2489, 2712))
    assert os_img.level_downsamples == (1.0, 4.000485597111555, 8.00251238191405)

    OpenSlideConverter().convert_image(svs_path, str(tmp_path))

    t = TileDBOpenSlide.from_group_uri(str(tmp_path))
    assert t.level_count == os_img.level_count
    assert t.dimensions == os_img.dimensions
    assert t.level_dimensions == os_img.level_dimensions
    assert t.level_downsamples == os_img.level_downsamples
    for factor, best_level in (32, 2), (3.9, 0), (4.1, 1), (2.9, 0), (1, 0):
        assert os_img.get_best_level_for_downsample(factor) == best_level
        assert t.get_best_level_for_downsample(factor) == best_level
    for level, dimensions in enumerate(os_img.level_dimensions):
        assert t.level_info[level] == LevelInfo(uri="", dimensions=dimensions)

    region = t.read_region(level=0, location=(100, 100), size=(300, 400))
    assert isinstance(region, np.ndarray)
    assert region.ndim == 3
    assert region.dtype == np.uint8
    img = PIL.Image.fromarray(region)
    assert img.size == (300, 400)
