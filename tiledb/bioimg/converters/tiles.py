import itertools as it
from typing import Iterator, Tuple

import numpy as np

import tiledb


def iter_tiles(domain: tiledb.Domain, scale: int = 1) -> Iterator[Tuple[slice, ...]]:
    """Generate all the non-overlapping tiles that cover the given TileDB domain."""
    return it.product(
        *map(iter_slices, map(dim_range, domain, [scale for _ in range(domain.ndim)]))
    )


def num_tiles(domain: tiledb.Domain, scale: int = 1) -> int:
    """Compute the number of non-overlapping tiles that cover the given TileDB domain."""
    n = 1
    for dim in domain:
        n *= len(dim_range(dim, scale=scale))
    return n


def dim_range(dim: tiledb.Dim, scale: int = 1) -> range:
    """Get the range of the given tiledb dimension with step equal to the dimension tile."""
    return range(
        int(dim.domain[0]), int(dim.domain[1]) + 1, dim.tile.astype(np.int64) * scale
    )


def iter_slices(r: range) -> Iterator[slice]:
    """
    Generate all the non-overlapping slices that cover the given range `r`,
    with each slice having length `r.step` (except possibly the last one).

    slice(r[0], r[1])
    slice(r[1], r[2])
    ...
    slice(r[n-2], r[n-1])
    slice(r[n-1], r.stop)
    """
    yield from it.starmap(slice, zip(r, r[1:]))
    yield slice(r[-1], r.stop)
