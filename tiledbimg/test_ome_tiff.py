import numpy as np
import tiledb
from open_slide import LevelInfo, TileDBOpenSlide

g_uri = "../data/CMU-1-Small-Region.tiledb"


def test_ome_tiff():
    # ToDo: We need to find better test data. This data has already been downsampled without preserving the original levels.

    # import openslide as osld
    # import tifffile
    # ometiff_uri = "../data/CMU-1-Small-Region.ome.tiff"
    # svstiff_uri = "../data/CMU-1-Small-Region.svs.tiff"
    # ometiff_img = tifffile.TiffFile(ometiff_uri)
    # os_img = osld.open_slide(svstiff_uri)

    t = TileDBOpenSlide.from_group_uri(g_uri)

    Reference_Schema = []

    Reference_Schema.append(
        tiledb.ArraySchema(
            domain=tiledb.Domain(
                *[
                    tiledb.Dim(name="X", domain=(0, 2219), tile=1024, dtype=np.uint64),
                    tiledb.Dim(name="Y", domain=(0, 2966), tile=1024, dtype=np.uint64),
                ]
            ),
            sparse=False,
            attrs=[
                tiledb.Attr(
                    name="rgb",
                    dtype=[("f0", "u1"), ("f1", "u1"), ("f2", "u1")],
                    var=False,
                    nullable=False,
                    filters=tiledb.FilterList([tiledb.ZstdFilter(level=0)]),
                )
            ],
            cell_order="row-major",
            tile_order="row-major",
            capacity=10000,
        )
    )

    Reference_Schema.append(
        tiledb.ArraySchema(
            domain=tiledb.Domain(
                *[
                    tiledb.Dim(name="X", domain=(0, 386), tile=387, dtype=np.uint64),
                    tiledb.Dim(name="Y", domain=(0, 462), tile=463, dtype=np.uint64),
                ]
            ),
            sparse=False,
            attrs=[
                tiledb.Attr(
                    name="rgb",
                    dtype=[("f0", "u1"), ("f1", "u1"), ("f2", "u1")],
                    var=False,
                    nullable=False,
                    filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)]),
                )
            ],
            cell_order="row-major",
            tile_order="row-major",
            capacity=10000,
        )
    )

    Reference_Schema.append(
        tiledb.ArraySchema(
            domain=tiledb.Domain(
                *[
                    tiledb.Dim(name="X", domain=(0, 1279), tile=1024, dtype=np.uint64),
                    tiledb.Dim(name="Y", domain=(0, 430), tile=431, dtype=np.uint64),
                ]
            ),
            sparse=False,
            attrs=[
                tiledb.Attr(
                    name="rgb",
                    dtype=[("f0", "u1"), ("f1", "u1"), ("f2", "u1")],
                    var=False,
                    nullable=False,
                    filters=tiledb.FilterList([tiledb.ZstdFilter(level=2)]),
                )
            ],
            cell_order="row-major",
            tile_order="row-major",
            capacity=10000,
        )
    )

    assert (
        t.level_info[0] == LevelInfo(abspath="", level=0, schema=Reference_Schema[0])
        and t.level_info[1]
        == LevelInfo(abspath="", level=1, schema=Reference_Schema[1])
        and t.level_info[2]
        == LevelInfo(abspath="", level=2, schema=Reference_Schema[2])
    )
    assert t.level_count == 3
    assert t.dimensions == (2220, 2967)
    assert t.level_dimensions == ((2220, 2967), (387, 463), (1280, 431))
    assert t.level_downsamples is None
