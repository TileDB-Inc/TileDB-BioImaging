import random

import pytest

import tiledb
from tests import get_schema
from tiledb.bioimg.helpers import open_bioimg
from tiledb.bioimg.openslide import TileDBOpenSlide

from .. import ATTR_NAME


class TestTileDBOpenSlide:
    def test(self, tmp_path):
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
                with open_bioimg(level_path, "w") as A:
                    A.meta["level"] = level
                G.add(level_path)

        with TileDBOpenSlide(group_path) as t:
            assert t.level_count == len(level_dimensions)
            assert t.level_dimensions == tuple(level_dimensions)

        with TileDBOpenSlide(group_path, attr=ATTR_NAME) as t:
            assert t.level_count == len(level_dimensions)
            assert t.level_dimensions == tuple(level_dimensions)

        with pytest.raises(KeyError) as e_info:
            _ = TileDBOpenSlide(group_path, attr="test_attr_name")
            assert "No attribute matching 'test_attr_name'" in str(e_info.value)
