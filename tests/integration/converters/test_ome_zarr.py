import numpy as np
import tiledb

from tests import get_CMU_1_SMALL_REGION_schemas, get_path
from tiledbimg.converters.ome_zarr import OMEZarrConverter
from tiledbimg.openslide import LevelInfo, TileDBOpenSlide


def test_ome_zarr_converter(tmp_path):
    OMEZarrConverter().convert_image(
        get_path("CMU-1-Small-Region.ome.zarr"), str(tmp_path)
    )
    schemas = get_CMU_1_SMALL_REGION_schemas()
    assert len(tiledb.Group(str(tmp_path))) == len(schemas)
    for i, schema in enumerate(schemas):
        with tiledb.open(str(tmp_path / f"l_{i}.tdb")) as A:
            assert A.schema == schema

    t = TileDBOpenSlide.from_group_uri(str(tmp_path))
    assert t.level_count == 3
    assert t.dimensions == (2220, 2967)
    assert t.level_dimensions == ((2220, 2967), (387, 463), (1280, 431))
    assert t.level_downsamples == (1.0, 6.0723207259698295, 4.30918285962877)
    for i in range(t.level_count):
        assert t.level_info[i] == LevelInfo(uri="", dimensions=schemas[i].shape[:2])
    region = t.read_region(level=0, location=(100, 100), size=(300, 400))
    assert isinstance(region, np.ndarray)
    assert region.dtype == np.uint8
    assert region.shape == (300, 400, 3)
