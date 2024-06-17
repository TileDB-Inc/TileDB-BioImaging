import itertools as it
from typing import Iterator, MutableSequence, Sequence, Tuple, Union

import tiledb


def iter_tiles(
    domain: Union[tiledb.Domain, Sequence[Tuple[int, int, int]]], scale: int = 1
) -> Iterator[Tuple[slice, ...]]:
    transformed_domain: MutableSequence[Tuple[int, int, int]] = []

    if isinstance(domain, tiledb.Domain):
        for dim in domain:
            transformed_domain.append(
                (int(dim.domain[0]), int(dim.domain[1]), int(dim.tile) * scale)
            )
    else:
        for dim in domain:
            transformed_domain.append((dim[0], dim[1], dim[2] * scale))

    """Generate all the non-overlapping tiles that cover the given TileDB domain."""
    return it.product(*map(iter_slices, map(dim_range, transformed_domain)))


def num_tiles(
    domain: Union[tiledb.Domain, Sequence[Tuple[int, int, int]]], scale: int = 1
) -> int:
    """Compute the number of non-overlapping tiles that cover the given TileDB domain."""
    transformed_domain: MutableSequence[Tuple[int, int, int]] = []

    if isinstance(domain, tiledb.Domain):
        for dim in domain:
            transformed_domain.append(
                (int(dim.domain[0]), int(dim.domain[1]), int(dim.tile) * scale)
            )
    else:
        for dim in domain:
            transformed_domain.append((dim[0], dim[1], dim[2] * scale))

    n = 1
    for dim in transformed_domain:
        n *= len(dim_range(dim))
    return n


def dim_range(dim: Tuple[int, int, int]) -> range:
    """Get the range of the given tiledb dimension with step equal to the dimension tile."""
    return range(dim[0], dim[1] + 1, dim[2])


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
