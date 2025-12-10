import enum
from typing import Literal


class Converters(enum.Enum):
    OMETIFF = enum.auto()
    OMEZARR = enum.auto()
    OSD = enum.auto()
    PNG = enum.auto()


DataProtocol = Literal["tiledbv2", "tiledbv3"]
