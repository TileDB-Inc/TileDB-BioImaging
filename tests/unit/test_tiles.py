import tiledb
from tiledb.bioimg.converters.tiles import dim_range, iter_slices, iter_tiles

domain = tiledb.Domain(
    tiledb.Dim("X", domain=(0, 9), tile=3),
    tiledb.Dim("Y", domain=(0, 14), tile=5),
    tiledb.Dim("Z", domain=(0, 6), tile=4),
    tiledb.Dim("C", domain=(0, 2), tile=3),
)


def test_dim_range():
    dims = list(domain)
    assert dim_range(dims[0]) == range(0, 10, 3)
    assert dim_range(dims[1]) == range(0, 15, 5)
    assert dim_range(dims[2]) == range(0, 7, 4)
    assert dim_range(dims[3]) == range(0, 3, 3)


def test_iter_slices():
    assert list(iter_slices(range(0, 9, 3))) == [slice(0, 3), slice(3, 6), slice(6, 9)]
    assert list(iter_slices(range(0, 10, 3))) == [
        slice(0, 3),
        slice(3, 6),
        slice(6, 9),
        slice(9, 10),
    ]


def test_iter_tiles():
    assert list(iter_tiles(domain)) == [
        (slice(0, 3), slice(0, 5), slice(0, 4), slice(0, 3)),
        (slice(0, 3), slice(0, 5), slice(4, 7), slice(0, 3)),
        (slice(0, 3), slice(5, 10), slice(0, 4), slice(0, 3)),
        (slice(0, 3), slice(5, 10), slice(4, 7), slice(0, 3)),
        (slice(0, 3), slice(10, 15), slice(0, 4), slice(0, 3)),
        (slice(0, 3), slice(10, 15), slice(4, 7), slice(0, 3)),
        (slice(3, 6), slice(0, 5), slice(0, 4), slice(0, 3)),
        (slice(3, 6), slice(0, 5), slice(4, 7), slice(0, 3)),
        (slice(3, 6), slice(5, 10), slice(0, 4), slice(0, 3)),
        (slice(3, 6), slice(5, 10), slice(4, 7), slice(0, 3)),
        (slice(3, 6), slice(10, 15), slice(0, 4), slice(0, 3)),
        (slice(3, 6), slice(10, 15), slice(4, 7), slice(0, 3)),
        (slice(6, 9), slice(0, 5), slice(0, 4), slice(0, 3)),
        (slice(6, 9), slice(0, 5), slice(4, 7), slice(0, 3)),
        (slice(6, 9), slice(5, 10), slice(0, 4), slice(0, 3)),
        (slice(6, 9), slice(5, 10), slice(4, 7), slice(0, 3)),
        (slice(6, 9), slice(10, 15), slice(0, 4), slice(0, 3)),
        (slice(6, 9), slice(10, 15), slice(4, 7), slice(0, 3)),
        (slice(9, 10), slice(0, 5), slice(0, 4), slice(0, 3)),
        (slice(9, 10), slice(0, 5), slice(4, 7), slice(0, 3)),
        (slice(9, 10), slice(5, 10), slice(0, 4), slice(0, 3)),
        (slice(9, 10), slice(5, 10), slice(4, 7), slice(0, 3)),
        (slice(9, 10), slice(10, 15), slice(0, 4), slice(0, 3)),
        (slice(9, 10), slice(10, 15), slice(4, 7), slice(0, 3)),
    ]
