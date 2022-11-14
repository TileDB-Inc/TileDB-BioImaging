import random

import tiledb
from tests import get_schema
from tiledb.bioimg.openslide import TileDBOpenSlide


class TestTileDBOpenSlide:
    def test_from_group_uri(self, tmp_path):
        def r():
            return random.randint(64, 4096)

        level_dimensions = [(r(), r()) for _ in range(random.randint(1, 10))]
        schemas = [get_schema(*dims) for dims in level_dimensions]
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
        assert tdb_os.level_count == len(level_dimensions)
        assert tdb_os.level_dimensions == tuple(level_dimensions)
