import json
from dataclasses import dataclass
from typing import List, Optional, Sequence

import tiledb

from ..converters.axes import Axes


@dataclass(frozen=True)
class ZoomLevelRecord:
    zoom_level: int
    image_level: int
    downsample: int
    width: int
    height: int
    namespace: str
    array_uri: str
    axes: str


def calculate_metadata(
    namespace: str, group_uri: str, config: Optional[tiledb.Config] = None
) -> Sequence[ZoomLevelRecord]:
    """
    Calculates the required metadata for the Tile Image viewer.

    :param namespace: the namespace of the group
    :param group_uri: path to the group
    :param config: the configuration for constructing the tileDB context
    """
    metadata: List[ZoomLevelRecord] = []

    context = None if config is None else tiledb.Ctx(config)
    group = tiledb.Group(group_uri, "r", ctx=context)
    level = 0

    for index, meta in enumerate(json.loads(group.meta["levels"])[::-1]):
        axes = Axes(meta["axes"])
        axes_mapper = axes.mapper(Axes("XYC"))
        shape = axes_mapper.map_shape(meta["shape"])

        zoom_factor = round(shape[0] / metadata[-1].width) if len(metadata) > 0 else 2

        if not ((zoom_factor & (zoom_factor - 1) == 0) and zoom_factor != 0):
            raise ValueError("Only power of two zoom factors are supported")

        for i in range(bin(zoom_factor - 1).count("1")):
            metadata.append(
                ZoomLevelRecord(
                    zoom_level=level,
                    image_level=index,
                    downsample=2 ** (bin(zoom_factor - 1).count("1") - i - 1),
                    width=shape[0],
                    height=shape[1],
                    namespace=namespace,
                    array_uri=group[meta["name"]].uri,
                    axes=meta["axes"],
                )
            )
            level += 1

    group.close()

    return metadata
