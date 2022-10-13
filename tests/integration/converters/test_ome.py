import os
import shutil

import numpy as np
import pytest
import tiledb

from tests import get_CMU_1_SMALL_REGION_schemas, get_path
from tiledbimg.converters.ome_tiff import OMETiffConverter
from tiledbimg.converters.ome_zarr import OMEZarrConverter
from tiledbimg.converters.openslide import OpenSlideConverter
from tiledbimg.openslide import LevelInfo, TileDBOpenSlide


@pytest.mark.parametrize(
    "uri,level_count",
    [
        ("CMU-1-Small-Region.svs.tiff.tdb", 1),
        ("CMU-1-Small-Region.ome.tiff.tdb", 3),
        ("CMU-1-Small-Region.ome.zarr.tdb", 3),
    ],
)
def test_ome(uri, level_count):
    # TODO: We need to find better test data.
    # This data has already been downsampled without preserving the original levels.
    t = TileDBOpenSlide.from_group_uri(get_path(uri))
    schemas = get_CMU_1_SMALL_REGION_schemas()

    assert t.level_count == level_count
    assert t.dimensions == (2220, 2967)
    assert t.level_dimensions == ((2220, 2967), (387, 463), (1280, 431))[:level_count]
    assert (
        t.level_downsamples == (1.0, 6.0723207259698295, 4.30918285962877)[:level_count]
    )

    for i in range(level_count):
        assert t.level_info[i] == LevelInfo(
            uri="", level=i, dimensions=schemas[i].shape[:2]
        )

    region = t.read_region(level=0, location=(100, 100), size=(300, 400))
    assert isinstance(region, np.ndarray)
    assert region.dtype == np.uint8
    assert region.shape == (300, 400, 3)


@pytest.mark.parametrize(
    "converter,path,level_count",
    [
        (OpenSlideConverter, "CMU-1-Small-Region.svs.tiff", 1),
        (OMETiffConverter, "CMU-1-Small-Region.ome.tiff", 3),
        (OMEZarrConverter, "CMU-1-Small-Region.ome.zarr", 3),
    ],
)
def test_ome_converter(tmp_path, converter, path, level_count):
    converter().convert_image(get_path(path), str(tmp_path))

    schemas = get_CMU_1_SMALL_REGION_schemas()[:level_count]
    assert len(tiledb.Group(str(tmp_path))) == len(schemas)
    for i, schema in enumerate(schemas):
        with tiledb.open(str(tmp_path / f"l_{i}.tdb")) as A:
            assert A.schema == schema


def test_ome_tiff_converter_different_dtypes(tmp_path):
    path = get_path("rand_uint16.ome.tiff")
    OMETiffConverter().convert_image(get_path(path), str(tmp_path))

    assert len(tiledb.Group(str(tmp_path))) == 2
    with tiledb.open(str(tmp_path / "l_0.tdb")) as A:
        assert A.schema.domain.dtype == np.uint32
        assert A.attr(0).dtype == np.uint16
    with tiledb.open(str(tmp_path / "l_1.tdb")) as A:
        assert A.schema.domain.dtype == np.uint16
        assert A.attr(0).dtype == np.uint8


@pytest.mark.parametrize("max_workers", [0, 1, 2])
def test_ome_tiff_converter_parallel(tmp_path, max_workers):
    src = get_path("CMU-1-Small-Region.ome.tiff")

    # Replicate image file in other location
    src2 = tmp_path / "src2"
    src2.mkdir()
    shutil.copy(src, src2)
    shutil.move(
        os.path.join(src2, "CMU-1-Small-Region.ome.tiff"),
        os.path.join(src2, "CMU-2-Small-Region.ome.tiff"),
    )

    image_list = [src, os.path.join(src2, "CMU-2-Small-Region.ome.tiff")]
    groups = tmp_path / "groups"
    groups.mkdir()
    if not os.path.exists(groups.as_uri()):
        OMETiffConverter().convert_images(
            image_list, groups.as_uri(), max_workers=max_workers
        )

    grp_1 = tiledb.Group(os.path.join(groups.as_uri(), "CMU-1-Small-Region.ome"), "r")
    assert grp_1.meta["original_filename"] == src
    grp_2 = tiledb.Group(os.path.join(groups.as_uri(), "CMU-2-Small-Region.ome"), "r")
    assert grp_2.meta["original_filename"] == os.path.join(
        src2, "CMU-2-Small-Region.ome.tiff"
    )
