from dataclasses import dataclass

import numpy as np
import tiledb


@dataclass(frozen=True)
class CMU_1_SMALL_REGION:
    _domains = {
        0: ((0, 2219, 1024), (0, 2966, 1024)),
        1: ((0, 386, 387), (0, 462, 463)),
        2: ((0, 1279, 1024), (0, 430, 431)),
    }
    _attr_dtype = [("f0", "u1"), ("f1", "u1"), ("f2", "u1")]

    def schemas(self):
        return [
            tiledb.ArraySchema(
                domain=tiledb.Domain(
                    *[
                        tiledb.Dim(
                            name="X",
                            domain=self._domains[elem_id][0][:2],
                            tile=self._domains[elem_id][0][-1],
                            dtype=np.uint64,
                        ),
                        tiledb.Dim(
                            name="Y",
                            domain=self._domains[elem_id][1][:2],
                            tile=self._domains[elem_id][1][-1],
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
            for elem_id in range(len(self._domains))
        ]
