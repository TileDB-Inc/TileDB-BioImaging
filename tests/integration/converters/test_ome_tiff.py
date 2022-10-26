import os
import shutil

import numpy as np
import PIL.Image
import pytest
import tiledb

from tests import get_CMU_1_SMALL_REGION_schemas, get_path
from tiledbimg.converters.ome_tiff import OMETiffConverter
from tiledbimg.openslide import LevelInfo, TileDBOpenSlide


def test_ome_tiff_converter(tmp_path):
    OMETiffConverter().convert_image(
        get_path("CMU-1-Small-Region.ome.tiff"), str(tmp_path)
    )
    schemas = get_CMU_1_SMALL_REGION_schemas(include_nested=True)
    assert len(tiledb.Group(str(tmp_path))) == len(schemas)
    for i, schema in enumerate(schemas):
        with tiledb.open(str(tmp_path / f"l_{i}.tdb")) as A:
            assert A.schema == schema

    t = TileDBOpenSlide.from_group_uri(str(tmp_path))
    assert t.level_count == 4
    assert t.dimensions == (2220, 2967)
    assert t.level_dimensions == ((2220, 2967), (574, 768), (387, 463), (1280, 431))
    assert t.level_downsamples == (
        1.0,
        3.865438534407666,
        6.0723207259698295,
        4.30918285962877,
    )
    for i in range(t.level_count):
        assert t.level_info[i] == LevelInfo(uri="", dimensions=schemas[i].shape[:-3:-1])

    region = t.read_region(level=0, location=(100, 100), size=(300, 400))
    assert isinstance(region, np.ndarray)
    assert region.ndim == 3
    assert region.dtype == np.uint8
    img = PIL.Image.fromarray(region)
    assert img.size == (300, 400)


def test_ome_tiff_converter_different_dtypes(tmp_path):
    path = get_path("rand_uint16.ome.tiff")
    OMETiffConverter().convert_image(get_path(path), str(tmp_path))

    assert len(tiledb.Group(str(tmp_path))) == 4
    with tiledb.open(str(tmp_path / "l_0.tdb")) as A:
        assert A.schema.domain.dtype == np.uint32
        assert A.attr(0).dtype == np.uint16
    with tiledb.open(str(tmp_path / "l_1.tdb")) as A:
        assert A.schema.domain.dtype == np.uint16
        assert A.attr(0).dtype == np.uint16
    with tiledb.open(str(tmp_path / "l_2.tdb")) as A:
        assert A.schema.domain.dtype == np.uint16
        assert A.attr(0).dtype == np.uint16
    with tiledb.open(str(tmp_path / "l_3.tdb")) as A:
        assert A.schema.domain.dtype == np.uint16
        assert A.attr(0).dtype == np.uint8


@pytest.mark.parametrize("max_workers,num_copies", [(0, 2), (2, 4), (None, 9)])
def test_ome_tiff_converter_parallel(tmp_path, max_workers, num_copies):
    input_paths = []
    src = get_path("CMU-1-Small-Region.ome.tiff")
    input_paths.append(src)
    for i in range(num_copies):
        dest = str(tmp_path / f"{i}-{os.path.basename(src)}")
        shutil.copy(src, dest)
        input_paths.append(dest)

    output_path = tmp_path / "converted"
    output_path.mkdir()
    OMETiffConverter().convert_images(input_paths, output_path, max_workers=max_workers)

    converted_dirs = list(map(str, output_path.glob("*")))
    assert len(converted_dirs) == len(input_paths)
    for converted_dir in converted_dirs:
        assert tiledb.object_type(converted_dir) == "group"
