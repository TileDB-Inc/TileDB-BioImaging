import json
from typing import Literal, Union, Sequence, Optional, Mapping, Any, Tuple

from attr import dataclass

SpaceUnit = Literal[
    'angstrom', 'attometer', 'centimeter', 'decimeter', 'exameter', 'femtometer', 'foot', 'gigameter', 'hectometer', 'inch', 'kilometer', 'megameter', 'meter', 'micrometer', 'mile', 'millimeter', 'nanometer', 'parsec', 'petameter', 'picometer', 'terameter', 'yard', 'yoctometer', 'yottameter', 'zeptometer', 'zettameter']
TimeUnit = Literal[
    'attosecond', 'centisecond', 'day', 'decisecond', 'exasecond', 'femtosecond', 'gigasecond', 'hectosecond', 'hour', 'kilosecond', 'megasecond', 'microsecond', 'millisecond', 'minute', 'nanosecond', 'petasecond', 'picosecond', 'second', 'terasecond', 'yoctosecond', 'yottasecond', 'zeptosecond', 'zettasecond']

spaceUnitSymbolMap = {
    "Å": 'angstrom',
    "am": 'attometer',
    "cm": 'centimeter',
    "dm": 'decimeter',
    "Em": 'exameter',
    "fm": 'femtometer',
    "ft": 'foot',
    "Gm": 'gigameter',
    "hm": 'hectometer',
    "in": 'inch',
    "km": 'kilometer',
    "Mm": 'megameter',
    "m": 'meter',
    "µm": 'micrometer',
    "mi.": 'mile',
    "mm": 'millimeter',
    "nm": 'nanometer',
    "pc": 'parsec',
    "Pm": 'petameter',
    "pm": 'picometer',
    "Tm": 'terameter',
    "yd": 'yard',
    "ym": 'yoctometer',
    "Ym": 'yottameter',
    "zm": 'zeptometer',
    "Zm": 'zettameter'
}

timeUnitSymbolMap = {
    "as": 'attosecond',
    "cs": 'centisecond',
    "d": 'day',
    "ds": 'decisecond',
    "Es": 'exasecond',
    "fs": 'femtosecond',
    "Gs": 'gigasecond',
    "hs": 'hectosecond',
    "h": 'hour',
    "ks": 'kilosecond',
    "Ms": 'megasecond',
    "µs": 'microsecond',
    "ms": 'millisecond',
    "min": 'minute',
    "ns": 'nanosecond',
    "Ps": 'petasecond',
    "ps": 'picosecond',
    "s": 'second',
    "Ts": 'terasecond',
    "ys": 'yoctosecond',
    "Ys": 'yottasecond',
    "zs": 'zeptosecond',
    "Zs": 'zettasecond'
}


class JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, NGFFLabelProperty):
            return {key: val for key, val in {**obj.__dict__, **obj.additionalMetadata} if val is not None}
        return {key: val for key, val in obj.__dict__ if val is not None}


@dataclass
class NGFFAxes:
    name: str
    type: Optional[Union[Literal['space', 'time', 'channel'], str]]
    unit: Optional[Union[SpaceUnit, TimeUnit]]


@dataclass
class NGFFCoordinateTransformation:
    type: Literal['identity', 'translation', 'scale']
    translation: Optional[Sequence[float]]
    scale: Optional[Sequence[float]]


@dataclass
class NGFFDataset:
    path: str
    coordinateTransformations: Sequence[NGFFCoordinateTransformation]


@dataclass
class NGFFMultiscale:
    version: str
    name: Optional[str]
    type: Optional[str]
    metadata: Optional[Mapping[str, Any]]
    axes: Sequence[NGFFAxes]
    datasets: Sequence[NGFFDataset]
    coordinateTransformations: Optional[Sequence[NGFFCoordinateTransformation]]


@dataclass
class NGFFLabelColor:
    labelValue: int
    rgba: Tuple[int, int, int, int]


@dataclass
class NGFFLabelProperty:
    labelValue: int
    additionalMetadata: Mapping[str, Any]


@dataclass
class NGFFLabelSource:
    image: str


@dataclass
class NGFFImageLabel:
    version: str
    colors: Optional[Sequence[NGFFLabelColor]]
    properties: Optional[Sequence[NGFFLabelProperty]]
    source: Optional[NGFFLabelSource]


@dataclass
class NGFFAcquisition:
    id: int
    name: Optional[str]
    maximumFieldCount: Optional[int]
    description: Optional[str]
    startTime: Optional[int]
    endTime: Optional[int]


@dataclass
class NGFFColumn:
    name: str


@dataclass
class NGFFRow:
    name: str


@dataclass
class NGFFPlateWell:
    path: str
    rowIndex: int
    columnIndex: int


@dataclass
class NGFFPlate:
    version: str
    columns: Sequence[NGFFColumn]
    rows: Sequence[NGFFRow]
    wells: Sequence[NGFFPlateWell]
    fieldCount: Optional[int]
    name: Optional[str]
    acquisitions: Optional[Sequence[NGFFAcquisition]]


@dataclass
class NGFFWellImage:
    path: str
    acquisition: Optional[int]


@dataclass
class NGFFWell:
    version: Optional[str]
    images: Sequence[NGFFWellImage]


class NGFFMetadata:
    axes: Sequence[NGFFAxes]
    coordinateTransformations: Optional[Sequence[NGFFCoordinateTransformation]]
    multiscales: Optional[Sequence[NGFFMultiscale]]
    labels: Optional[Sequence[str]]

    # TODO How should we store NGFFImageLabels

    @classmethod
    def from_ome_tiff(cls, ome_metadata: Union[dict[str, Any], dict]):
        metadata = cls()

        # If invalid OME metadata return empty NGFF metadata
        if 'OME' not in ome_metadata:
            return metadata

        ome_images = ome_metadata.get('OME', {}).get('Image', [])
        if not ome_images:
            return metadata

        ome_pixels = ome_images[0].get('Pixels', {}) if isinstance(ome_images, list) else ome_images.get('Pixels', {})

        # Create 'axes' metadata field
        if 'DimensionOrder' in ome_pixels:
            axes = []
            for axis in ome_pixels.get('DimensionOrder', ''):
                if axis in ['X', 'Y', 'Z']:
                    axes.append(NGFFAxes(name=axis, type='space',
                                         unit=spaceUnitSymbolMap.get(ome_pixels.get(f'PhysicalSize{axis}Unit', "µm"))))
                elif axis == 'C':
                    axes.append(NGFFAxes(name=axis, type='channel', unit=None))
                elif axis == 'T':
                    axes.append(NGFFAxes(name=axis, type='time',
                                         unit=timeUnitSymbolMap.get(ome_pixels.get(f'TimeIncrementUnit', "s"))))
                else:
                    axes.append(NGFFAxes(name=axis, type=None, unit=None))
            metadata.axes = axes

        return metadata
