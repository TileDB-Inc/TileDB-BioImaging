from dataclasses import dataclass

import numpy as np
import tiledb


@dataclass(frozen=True)
class CMU_1_SMALL_REGION:
    _dom = {
        0: ((0, 2219), (0, 2966)),
        1: ((0, 386), (0, 462)),
        2: ((0, 1279), (0, 430)),
    }
    _attr_dtype = [("f0", "u1"), ("f1", "u1"), ("f2", "u1")]
    _tile = {0: (1024, 1024), 1: (387, 463), 2: (1024, 431)}

    def schema(self):
        return [
            tiledb.ArraySchema(
                domain=tiledb.Domain(
                    *[
                        tiledb.Dim(
                            name="X",
                            domain=self._dom[elem_id][0],
                            tile=self._tile[elem_id][0],
                            dtype=np.uint64,
                        ),
                        tiledb.Dim(
                            name="Y",
                            domain=self._dom[elem_id][1],
                            tile=self._tile[elem_id][1],
                            dtype=np.uint64,
                        ),
                    ]
                ),
                sparse=False,
                attrs=[
                    tiledb.Attr(
                        name="rgb",
                        dtype=self._attr_dtype,
                        var=False,
                        nullable=False,
                        filters=tiledb.FilterList([tiledb.ZstdFilter(level=0)]),
                    )
                ],
                cell_order="row-major",
                tile_order="row-major",
                capacity=10000,
            )
            for elem_id in range(0, 3)
        ]
