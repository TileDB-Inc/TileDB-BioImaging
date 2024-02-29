import json
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import tifffile
from tifffile import TiffFile
from typing_extensions import Literal, Self

SpaceUnit = Literal[
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]
TimeUnit = Literal[
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
]

spaceUnitSymbolMap: Mapping[str, SpaceUnit] = {
    "Å": "angstrom",
    "am": "attometer",
    "cm": "centimeter",
    "dm": "decimeter",
    "Em": "exameter",
    "fm": "femtometer",
    "ft": "foot",
    "Gm": "gigameter",
    "hm": "hectometer",
    "in": "inch",
    "km": "kilometer",
    "Mm": "megameter",
    "m": "meter",
    "µm": "micrometer",
    "mi.": "mile",
    "mm": "millimeter",
    "nm": "nanometer",
    "pc": "parsec",
    "Pm": "petameter",
    "pm": "picometer",
    "Tm": "terameter",
    "yd": "yard",
    "ym": "yoctometer",
    "Ym": "yottameter",
    "zm": "zeptometer",
    "Zm": "zettameter",
}

timeUnitSymbolMap: Mapping[str, TimeUnit] = {
    "as": "attosecond",
    "cs": "centisecond",
    "d": "day",
    "ds": "decisecond",
    "Es": "exasecond",
    "fs": "femtosecond",
    "Gs": "gigasecond",
    "hs": "hectosecond",
    "h": "hour",
    "ks": "kilosecond",
    "Ms": "megasecond",
    "µs": "microsecond",
    "ms": "millisecond",
    "min": "minute",
    "ns": "nanosecond",
    "Ps": "petasecond",
    "ps": "picosecond",
    "s": "second",
    "Ts": "terasecond",
    "ys": "yoctosecond",
    "Ys": "yottasecond",
    "zs": "zeptosecond",
    "Zs": "zettasecond",
}


class JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, NGFFLabelProperty):
            return {
                key: val
                for key, val in {
                    **obj.__dict__,
                    **(obj.additionalMetadata if obj.additionalMetadata else {}),
                }.items()
                if val is not None
            }
        return {key: val for key, val in obj.__dict__ if val is not None}


class NGFFAxes:
    def __init__(
        self,
        name: str,
        type: Optional[Union[Literal["space", "time", "channel"], str]] = None,
        unit: Optional[Union[SpaceUnit, TimeUnit]] = None,
    ):
        self.name = name
        self.type = type
        self.unit = unit

    name: str
    type: Optional[Union[Literal["space", "time", "channel"], str]]
    unit: Optional[Union[SpaceUnit, TimeUnit]]


class NGFFCoordinateTransformation:
    def __init__(
        self,
        type: Literal["identity", "translation", "scale"],
        translation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
    ):
        self.type = type
        self.translation = translation
        self.scale = scale

    type: Literal["identity", "translation", "scale"]
    translation: Optional[Sequence[float]]
    scale: Optional[Sequence[float]]


class NGFFDataset:
    def __init__(
        self,
        path: str,
        coordinateTransformations: Sequence[NGFFCoordinateTransformation],
    ):
        self.path = path
        self.coordinateTransformations = coordinateTransformations

    path: str
    coordinateTransformations: Sequence[NGFFCoordinateTransformation]


class NGFFMultiscale:
    def __init__(
        self,
        version: str,
        axes: Sequence[NGFFAxes],
        datasets: Sequence[NGFFDataset],
        name: Optional[str] = None,
        type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        coordinateTransformations: Optional[
            Sequence[NGFFCoordinateTransformation]
        ] = None,
    ):
        self.version = version
        self.name = name
        self.type = type
        self.metadata = metadata
        self.axes = axes
        self.datasets = datasets
        self.coordinateTransformations = coordinateTransformations

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


class NGFFLabelProperty:
    def __init__(
        self, labelValue: int, additionalMetadata: Optional[Mapping[str, Any]] = None
    ):
        self.labelValue = labelValue
        self.additionalMetadata = additionalMetadata

    labelValue: int
    additionalMetadata: Optional[Mapping[str, Any]]


@dataclass
class NGFFLabelSource:
    image: str


class NGFFImageLabel:
    def __init__(
        self,
        version: str,
        colors: Optional[Sequence[NGFFLabelColor]] = None,
        properties: Optional[Sequence[NGFFLabelProperty]] = None,
        source: Optional[NGFFLabelSource] = None,
    ):
        self.version = version
        self.colors = colors
        self.properties = properties
        self.source = source

    version: str
    colors: Optional[Sequence[NGFFLabelColor]]
    properties: Optional[Sequence[NGFFLabelProperty]]
    source: Optional[NGFFLabelSource]


class NGFFAcquisition:
    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        maximumFieldCount: Optional[int] = None,
        description: Optional[str] = None,
        startTime: Optional[int] = None,
        endTime: Optional[int] = None,
    ):
        self.id = id
        self.name = name
        self.maximumFieldCount = maximumFieldCount
        self.description = description
        self.startTime = startTime
        self.endTime = endTime

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


class NGFFPlate:
    def __init__(
        self,
        version: str,
        columns: Sequence[NGFFColumn],
        rows: Sequence[NGFFRow],
        wells: Sequence[NGFFPlateWell],
        fieldCount: Optional[int] = None,
        name: Optional[str] = None,
        acquisitions: Optional[Sequence[NGFFAcquisition]] = None,
    ):
        self.version = version
        self.columns = columns
        self.rows = rows
        self.wells = wells
        self.fieldCount = fieldCount
        self.name = name
        self.acquisitions = acquisitions

    version: str
    columns: Sequence[NGFFColumn]
    rows: Sequence[NGFFRow]
    wells: Sequence[NGFFPlateWell]
    fieldCount: Optional[int]
    name: Optional[str]
    acquisitions: Optional[Sequence[NGFFAcquisition]]

    @classmethod
    def from_ome_tiff(
        cls, ome_metadata: Mapping[str, Any]
    ) -> Union[Mapping[str, Self], None]:
        ome_plates = ome_metadata.get("OME", {}).get("Plate", [])
        ome_plates = [ome_plates] if not isinstance(ome_plates, list) else ome_plates
        plates: MutableMapping[str, Self] = {}

        if not len(ome_plates):
            return None

        for ome_plate in ome_plates:
            wells: MutableSequence[NGFFPlateWell] = []
            acquisitions: MutableSequence[NGFFAcquisition] = []

            row_naming: Literal["number", "letter"] = ome_plate.get(
                "RowNamingConvention", "number"
            )
            column_naming: Literal["number", "letter"] = ome_plate.get(
                "ColumnNamingConvention", "number"
            )

            ome_acquisitions = ome_plate.get("PlateAcquisition", [])
            ome_acquisitions = (
                [ome_acquisitions]
                if not isinstance(ome_acquisitions, list)
                else ome_acquisitions
            )
            for ome_acquisition in ome_acquisitions:
                start_time = (
                    int(
                        datetime.fromisoformat(
                            ome_acquisition.get("StartTime")
                        ).timestamp()
                    )
                    if "StartTime" in ome_acquisition
                    else None
                )
                end_time = (
                    int(
                        datetime.fromisoformat(
                            ome_acquisition.get("EndTime")
                        ).timestamp()
                    )
                    if "EndTime" in ome_acquisition
                    else None
                )
                acquisitions.append(
                    NGFFAcquisition(
                        id=ome_acquisition.get("ID"),
                        name=ome_acquisition.get("Name"),
                        description=ome_acquisition.get("Description"),
                        maximumFieldCount=ome_acquisition.get("MaximumFieldCount"),
                        startTime=start_time,
                        endTime=end_time,
                    )
                )

            number_of_rows = 1
            number_of_columns = 1
            ome_wells = ome_plate.get("Well", [])
            ome_wells = [ome_wells] if not isinstance(ome_wells, list) else ome_wells

            for ome_well in ome_wells:
                number_of_rows = max(ome_well.get("Row") + 1, number_of_rows)
                number_of_columns = max(ome_well.get("Column") + 1, number_of_columns)
                wells.append(
                    NGFFPlateWell(
                        path=f'{format_number(ome_well.get("Row"), row_naming)}/{format_number(ome_well.get("Column"), column_naming)}',
                        rowIndex=ome_well.get("Row"),
                        columnIndex=ome_well.get("Column"),
                    )
                )
            plates.setdefault(
                ome_plate.get("ID"),
                cls(
                    version="0.5-dev",
                    columns=[
                        NGFFColumn(format_number(idx, column_naming))
                        for idx in range(number_of_columns)
                    ],
                    rows=[
                        NGFFRow(format_number(idx, row_naming))
                        for idx in range(number_of_rows)
                    ],
                    wells=wells,
                    acquisitions=acquisitions if len(acquisitions) else None,
                    name=ome_plate.get("Name"),
                ),
            )

        return plates


class NGFFWellImage:
    def __init__(self, path: str, acquisition: Optional[int] = None):
        self.path = path
        self.acquisition = acquisition

    path: str
    acquisition: Optional[int]


class NGFFWell:
    def __init__(self, images: Sequence[NGFFWellImage], version: Optional[str] = None):
        self.version = version
        self.images = images

    version: Optional[str]
    images: Sequence[NGFFWellImage]

    @classmethod
    def from_ome_tiff(
        cls, ome_metadata: Mapping[str, Any]
    ) -> Optional[Mapping[str, Mapping[Tuple[int, int], Self]]]:
        ome_images = ome_metadata.get("OME", {}).get("Image", [])
        ome_images = [ome_images] if not isinstance(ome_images, list) else ome_images
        ome_plates = ome_metadata.get("OME", {}).get("Plate", [])
        ome_plates = [ome_plates] if not isinstance(ome_plates, list) else ome_plates

        wells: MutableMapping[str, MutableMapping[Tuple[int, int], Self]] = {}

        if not len(ome_plates) or not len(ome_images):
            return None

        for ome_plate in ome_plates:
            ome_acquisitions = ome_plate.get("PlateAcquisition", [])
            ome_acquisitions = (
                [ome_acquisitions]
                if not isinstance(ome_acquisitions, list)
                else ome_acquisitions
            )
            ome_wells = ome_plate.get("Well", [])
            ome_wells = [ome_wells] if not isinstance(ome_wells, list) else ome_wells

            if not len(ome_plate) or not len(ome_wells):
                continue

            image_name_map: MutableMapping[str, str] = {}
            for image in ome_images:
                image_name_map.setdefault(
                    image.get("ID"), image.get("Name", image.get("ID"))
                )

            sample_acquisition_map: MutableMapping[str, int] = {}
            for idx, acquisition in enumerate(ome_acquisitions):
                for sample in acquisition.get("WellSampleRef", []):
                    sample_acquisition_map.setdefault(sample.get("ID"), idx)

            wells.setdefault(ome_plate.get("ID"), {})

            for well in ome_wells:
                images: MutableSequence[NGFFWellImage] = []
                ome_samples = well.get("WellSample", [])
                ome_samples = (
                    [ome_samples] if not isinstance(ome_samples, list) else ome_samples
                )
                for sample in ome_samples:
                    images.append(
                        NGFFWellImage(
                            path=image_name_map.get(
                                sample.get("ImageRef", {}).get("ID"), ""
                            ),
                            acquisition=sample_acquisition_map.get(sample.get("ID")),
                        )
                    )
                wells.get(ome_plate.get("ID"), {}).setdefault(
                    (int(well.get("Row")), int(well.get("Column"))),
                    cls(images=images, version="0.5-dev"),
                )

        return wells


class NGFFMetadata:
    def __init__(
        self,
        axes: Sequence[NGFFAxes],
        coordinateTransformations: Optional[
            Sequence[NGFFCoordinateTransformation]
        ] = None,
        multiscales: Optional[Sequence[NGFFMultiscale]] = None,
        plate: Optional[Mapping[str, NGFFPlate]] = None,
        wells: Optional[Mapping[str, Mapping[Tuple[int, int], NGFFWell]]] = None,
    ):
        self.axes = axes
        self.coordinateTransformations = coordinateTransformations
        self.multiscales = multiscales
        self.plate = plate
        self.wells = wells

    axes: Sequence[NGFFAxes]
    coordinateTransformations: Optional[Sequence[NGFFCoordinateTransformation]]
    multiscales: Optional[Sequence[NGFFMultiscale]]
    labels: Optional[Sequence[str]]
    # Image Labels are stored at the label image level
    imageLabels: Optional[Sequence[NGFFImageLabel]]

    # Plate metadata should be written at the group level of each plate
    plate: Optional[Mapping[str, NGFFPlate]]

    # Wells metadata should be written at the group level of each well.
    # Each well is identified by a tuple (row, column)
    wells: Optional[Mapping[str, Mapping[Tuple[int, int], NGFFWell]]]

    @classmethod
    def from_ome_tiff(cls, tiff: TiffFile) -> Union[Self, None]:
        multiscales: MutableSequence[NGFFMultiscale] = []
        ome_metadata = tifffile.xml2dict(tiff.ome_metadata) if tiff.ome_metadata else {}

        if "OME" not in ome_metadata:
            return None

        ome_images = ome_metadata.get("OME", {}).get("Image", [])
        ome_images = [ome_images] if not isinstance(ome_images, list) else ome_images

        if not len(ome_images):
            return None

        # Step 1: Indentify all axes of the image. Special care must be taken for modulo datasets
        # where multiple axes are squashed in TCZ dimensions.
        xml_annotations = (
            ome_metadata.get("OME", {})
            .get("StructuredAnnotations", {})
            .get("XMLAnnotation", {})
        )

        if not isinstance(xml_annotations, list):
            xml_annotations = [xml_annotations]

        ome_modulo = {}
        for annotation in (
            raw_annotation.get("Value", {}) for raw_annotation in xml_annotations
        ):
            if "Modulo" in annotation:
                ome_modulo = annotation.get("Modulo", {})

        additional_axes = dict()
        for modulo_key in ["ModuloAlongZ", "ModuloAlongT", "ModuloAlongC"]:
            if modulo_key not in ome_modulo:
                continue

            modulo = ome_modulo.get(modulo_key, {})
            axis = NGFFAxes(
                name=modulo_key,
                type=modulo.get("Type", None),
                unit=modulo.get("Unit", None),
            )
            axis_size = (
                len(modulo.get("Label", []))
                if "Label" in modulo
                else (modulo.get("End") - modulo.get("Start")) / modulo.get("Step", 1)
                + 1
            )
            additional_axes[modulo_key] = (axis, axis_size)

        ome_pixels = ome_images[0].get("Pixels", {})
        canonical_axes = [
            "T",
            "ModuloAlongT",
            "ModuloAlongC",
            "ModuloAlongZ",
            "C",
            "Z",
            "Y",
            "X",
        ]
        # Create 'axes' metadata field
        axes = []
        for canonical_axis in canonical_axes:
            if canonical_axis in ["X", "Y", "Z"]:
                _, modulo_size = additional_axes.get(
                    f"ModuloAlong{canonical_axis}", (None, 1)
                )
                if ome_pixels.get(f"Size{canonical_axis}") > modulo_size:
                    axes.append(
                        NGFFAxes(
                            name=canonical_axis,
                            type="space",
                            unit=spaceUnitSymbolMap.get(
                                ome_pixels.get(
                                    f"PhysicalSize{canonical_axis}Unit", "µm"
                                )
                            ),
                        )
                    )
            elif canonical_axis == "C":
                axes.append(NGFFAxes(name=canonical_axis, type="channel"))
            elif canonical_axis == "T":
                _, modulo_size = additional_axes.get("ModuloAlongT", (None, 1))
                if ome_pixels.get("SizeT") > modulo_size:
                    axes.append(
                        NGFFAxes(
                            name=canonical_axis,
                            type="time",
                            unit=timeUnitSymbolMap.get(
                                ome_pixels.get("TimeIncrementUnit", "s")
                            ),
                        )
                    )
            elif canonical_axis in additional_axes:
                axes.append(additional_axes.get(canonical_axis, [])[0])

        # Create 'multiscales' metadata field
        for idx, series in enumerate(tiff.series):
            ome_pixels = ome_images[idx].get("Pixels", {})
            datasets: MutableSequence[NGFFDataset] = []
            x_index, y_index = series.levels[0].axes.index("X"), series.levels[
                0
            ].axes.index("Y")
            base_size = {
                "X": series.levels[0].shape[x_index],
                "Y": series.levels[0].shape[y_index],
            }

            # Calculate axis using the base image
            level_shape = list(series.levels[0].shape)

            # We need to map each modulo axis to its axis symbol
            # Step 1: Iterate the dimension order
            axes_order = []
            for dim in reversed(ome_pixels.get("DimensionOrder", "")):
                size = ome_pixels.get(f"Size{dim}", 1)

                # If dimension size is 1 then the axis is skipped
                if size == 1:
                    continue

                if dim in series.levels[0].axes:
                    # If the axis appear in the level axes then we add the axis
                    axes_order.append(dim)

                    # If the length of the axis does not match its size then there must be a modulo axis
                    if size != level_shape[0]:
                        axes_order.append(f"ModuloAlong{dim}")
                        level_shape.pop(0)
                    level_shape.pop(0)
                else:
                    axes_order.append(f"ModuloAlong{dim}")
                    level_shape.pop(0)

            if "C" not in axes_order:
                axes_order.append("C")

            for idx, level in enumerate(series.levels):
                if len(axes_order) != len(level.shape):
                    level_shape = list(level.shape) + [1]
                else:
                    level_shape = list(level.shape)

                # Step 2: Calculate scale information for each axis after transpose
                scale = []
                for axis in axes:
                    size = level_shape[axes_order.index(axis.name)]

                    if axis.name in ["X", "Y"]:
                        scale.append(
                            ome_pixels.get(f"PhysicalSize{axis.name}", 1)
                            * base_size.get(axis.name, size)
                            / size
                        )
                    else:
                        scale.append(1)

                datasets.append(
                    NGFFDataset(
                        level.name, [NGFFCoordinateTransformation("scale", scale)]
                    )
                )
            scale = []
            for axis in axes:
                if axis.name == "T":
                    scale.append(ome_pixels.get("TimeIncrement", 1))
                elif axis.name == "Z":
                    scale.append(ome_pixels.get("PhysicalSizeZ", 1))
                else:
                    scale.append(1)
            coordinateTransformation = (
                [NGFFCoordinateTransformation(type="scale", scale=scale)]
                if not all(factor == 1 for factor in scale)
                else None
            )
            multiscales.append(
                NGFFMultiscale(
                    version="0.5-dev",
                    name=series.name,
                    type=None,
                    metadata=None,
                    axes=axes,
                    datasets=datasets,
                    coordinateTransformations=coordinateTransformation,
                )
            )

        return cls(
            axes=axes,
            multiscales=multiscales,
            plate=NGFFPlate.from_ome_tiff(ome_metadata),
            wells=NGFFWell.from_ome_tiff(ome_metadata),
        )


def format_number(value: int, naming_convention: Literal["number", "letter"]) -> str:
    if naming_convention == "number":
        return str(value)

    value += 1

    result = ""
    while value > 0:
        result = chr(ord("A") + (value - 1) % 26) + result
        value = int((value - (value - 1) % 26) / 26)

    return result
