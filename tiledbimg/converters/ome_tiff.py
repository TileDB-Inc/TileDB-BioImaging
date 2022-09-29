import tifffile
import tiledb

from .base import ImageConverter


class OMETiffConverter(ImageConverter):
    """Converter of Tiff-supported images to TileDB Groups of Arrays"""

    def convert_image(
        self, input_path: str, output_group_path: str, level_min: int = 0
    ) -> None:
        tiledb.group_create(output_group_path)
        tiff_series = tifffile.TiffFile(input_path).series
        uris = []
        for level in range(level_min, len(tiff_series)):
            data = tiff_series[level].asarray().swapaxes(0, 2)
            uris.append(self._write_level(output_group_path, level, data))
        # TODO: level_downsamples
        self._write_metadata(output_group_path, input_path, uris=uris)
