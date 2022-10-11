import os

import pytest
import tiledb

from tests import check_level_info, get_CMU_1_SMALL_REGION_schemas
from tiledbimg.openslide import LevelInfo


class TestLevelInfo:
    def test_from_array(self, tmp_path):
        test_data_schemas = get_CMU_1_SMALL_REGION_schemas()
        tiledb.Array.create(os.path.join(tmp_path, "test.tdb"), test_data_schemas[0])
        with tiledb.open(os.path.join(tmp_path, "test.tdb"), "r") as A:
            for level in range(42):
                check_level_info(level, LevelInfo.from_array(A, level))

            with pytest.raises(ValueError) as excinfo:
                LevelInfo.from_array(A)
            assert "Invalid level uri" in str(excinfo)
