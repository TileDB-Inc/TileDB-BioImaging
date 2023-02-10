import json
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

import tiledb

from .converters.axes import Axes


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


def iter_zoom_level_records(
    namespace: str, group_uri: str, config: Optional[tiledb.Config] = None
) -> Iterator[ZoomLevelRecord]:
    """
    Calculates the required metadata for the Tile Image viewer.

    :param namespace: the namespace of the group
    :param group_uri: path to the group
    :param config: the configuration for constructing the tileDB context
    """
    with tiledb.Group(group_uri, mode="r", ctx=tiledb.Ctx(config)) as group:
        zoom_level = 0
        previous_level_width = 0
        for meta in reversed(json.loads(group.meta["levels"])):
            axes = Axes(meta["axes"])
            axes_mapper = axes.mapper(Axes("XYC"))
            width, height, channel = axes_mapper.map_shape(meta["shape"])

            if previous_level_width > 0:
                zoom_factor = round(width / previous_level_width)
            else:
                zoom_factor = 2

            previous_level_width = width

            n = np.log2(zoom_factor)
            if n != n.astype(int):
                raise ValueError("Only power of two zoom factors are supported")

            for downsample in reversed(2 ** np.arange(n.astype(int))):
                yield ZoomLevelRecord(
                    zoom_level=zoom_level,
                    image_level=meta["level"],
                    downsample=downsample,
                    width=width,
                    height=height,
                    namespace=namespace,
                    array_uri=group[meta["name"]].uri,
                    axes=meta["axes"],
                )
                zoom_level += 1
