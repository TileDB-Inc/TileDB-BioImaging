import itertools
from typing import MutableMapping, Optional, Sequence, Tuple

import tifffile


def get_pages_memory_order(
    tiff: tifffile.TiffFile,
) -> MutableMapping[int, Tuple[str, ...]]:
    """
    Return the stored axes memory order for each page of the first tiff series
    """

    memory_order: MutableMapping[int, Tuple[str, ...]] = {}

    for level in tiff.series[0].levels:
        for page in level.pages:
            # Order the chunk indices based on their memory location within the page
            chunks = [
                x for _, x in sorted(zip(page.dataoffsets, get_chunks(page.chunked)))
            ]

            # Get the axes order as defined by the memory location of the indices
            memory_order[page.hash] = get_page_order(chunks, page.chunked, page.axes)

    return memory_order


def get_chunks(
    shape: Tuple[int, ...], permutation: Optional[Sequence[int]] = None
) -> Sequence[Sequence[int]]:
    """
    Calculate the list of all chuck indices for a given shape

    :param shape: The number of chunks in each dimension
    :param permutation: If specified, change the order in which the dimensions are iterated based on the given permutation. Default None.
    """
    if permutation is None:
        permutation = [i for i in range(len(shape))]

    return [
        [coord[p] for p in permutation]
        for coord in itertools.product(*[[j for j in range(i)] for i in shape])
    ]


def get_page_order(
    chunks: Sequence[Sequence[int]], shape: Tuple[int, ...], axes: str
) -> Tuple[str, ...]:
    """
    Calculate the memory page order by comparing all possible axes permutations

    :param chunks: The memory ordered chuck indices
    :param shape: The number of chucks per dimension
    :param axes: The image axes
    """
    for perm_shape, perm_axes in zip(
        itertools.permutations(range(len(shape))), itertools.permutations(axes)
    ):
        if get_chunks(tuple(shape[p] for p in perm_shape), perm_shape) == chunks:
            return perm_axes

    return tuple()
