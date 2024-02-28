from typing import Union, Sequence, Any
from .constants import spaceUnitSymbolMap, timeUnitSymbolMap
from .converters.axes import NGFFAxes
import tifffile
from tifffile import TiffFile
from typing_extensions import Self


class NGFFMetadata:
    axes: Sequence[NGFFAxes]

    @classmethod
    def from_ome_tiff(cls, tiff: TiffFile) -> Union[Self, None]:
        ome_metadata = tifffile.xml2dict(tiff.ome_metadata) if tiff.ome_metadata else {}
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
