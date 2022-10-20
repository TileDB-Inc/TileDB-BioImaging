import tiledb

from tests import get_CMU_1_SMALL_REGION_schemas
from tiledbimg.openslide import LevelInfo, TileDBOpenSlide


class TestTileDBOpenSlide:
    def test_from_group_uri(self, tmp_path):
        schemas = get_CMU_1_SMALL_REGION_schemas()
        group_path = str(tmp_path)
        tiledb.Group.create(group_path)
        with tiledb.Group(group_path, "w") as G:
            for level, schema in enumerate(schemas):
                level_path = str(tmp_path / f"l_{level}.tdb")
                tiledb.Array.create(level_path, schema)
                with tiledb.open(level_path, "w") as A:
                    A.meta["level"] = level
                G.add(level_path)

        tdb_os = TileDBOpenSlide.from_group_uri(group_path)
        assert tdb_os.level_info == (
            LevelInfo(str(tmp_path / "l_0.tdb"), dimensions=(2220, 2967)),
            LevelInfo(str(tmp_path / "l_1.tdb"), dimensions=(387, 463)),
            LevelInfo(str(tmp_path / "l_2.tdb"), dimensions=(1280, 431)),
        )
