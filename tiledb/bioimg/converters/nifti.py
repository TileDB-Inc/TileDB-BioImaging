from __future__ import annotations

import base64
import json
import logging
from functools import partial
from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import nibabel as nib
import numpy as np
from nibabel import Nifti1Image
from nibabel.analyze import _dtdefs
from numpy._typing import NDArray

from tiledb import VFS, Config, Ctx
from tiledb.cc import WebpInputFormat
from tiledb.highlevel import _get_ctx

from ..helpers import get_logger_wrapper, iter_color
from .axes import Axes
from .base import ImageConverterMixin


# Function to find and return the third value based on the first value
def get_dtype_from_code(dtype_code: int) -> Optional[np.dtype]:
    for item in _dtdefs:
        if item[0] == dtype_code:  # Check if the first value matches the input code
            return item[2]  # Return the third value (dtype)
    return None  # Return None if the code is not found


class NiftiReader:
    _logger: logging.Logger

    def __init__(
        self,
        input_path: str,
        logger: Optional[logging.Logger] = None,
        *,
        source_config: Optional[Config] = None,
        source_ctx: Optional[Ctx] = None,
        dest_config: Optional[Config] = None,
        dest_ctx: Optional[Ctx] = None,
    ):
        self._logger = get_logger_wrapper(False) if not logger else logger
        self._input_path = input_path
        self._source_ctx = _get_ctx(source_ctx, source_config)
        self._source_cfg = self._source_ctx.config()
        self._dest_ctx = _get_ctx(dest_ctx, dest_config)
        self._dest_cfg = self._dest_ctx.config()
        self._vfs = VFS(config=self._source_cfg, ctx=self._source_ctx)
        self._vfs_fh = self._vfs.open(input_path, mode="rb")
        self._nib_image = Nifti1Image.from_stream(self._vfs_fh)
        self._metadata: Dict[str, Any] = self._serialize_header(
            self.nifti1_hdr_2_dict()
        )
        self._binary_header = base64.b64encode(
            self._nib_image.header.binaryblock
        ).decode("utf-8")
        self._mode = (
            "".join(self._nib_image.dataobj.dtype.names)
            if self._nib_image.dataobj.dtype.names is not None
            else ""
        )

    def __enter__(self) -> NiftiReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._vfs.close(file=self._vfs_fh)

    @property
    def source_ctx(self) -> Ctx:
        return self._source_ctx

    @property
    def dest_ctx(self) -> Ctx:
        return self._dest_ctx

    @property
    def logger(self) -> Optional[logging.Logger]:
        return self._logger

    @property
    def group_metadata(self) -> Dict[str, Any]:
        writer_kwargs = dict(
            metadata=self._metadata,
            binaryblock=self._binary_header,
            slope=self._nib_image.dataobj.slope,
            inter=self._nib_image.dataobj.inter,
        )
        self._logger.debug(f"Group metadata: {writer_kwargs}")
        return {"json_write_kwargs": json.dumps(writer_kwargs)}

    @property
    def image_metadata(self) -> Dict[str, Any]:
        self._logger.debug(f"Image metadata: {self._metadata}")
        color_generator = iter_color(np.dtype(np.uint8), len(self.channels))

        channels = []
        for idx, channel in enumerate(self.channels):
            channel_metadata = {
                "id": f"{idx}",
                "name": f"Channel {idx}",
                "color": (next(color_generator)),
            }
            channels.append(channel_metadata)
        self._metadata["channels"] = channels
        return self._metadata

    @property
    def axes(self) -> Axes:
        header_dict = self.nifti1_hdr_2_dict()
        # The 0-index holds information about the number of dimensions
        # according the spec https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
        dims_number = header_dict["dim"][0]
        if dims_number == 4:
            # According to standard the 4th dimension corresponds to 'T' time
            # but in special cases can be degnerate into channels
            if header_dict["dim"][dims_number] == 1:
                # The time dimension does not correspond to time
                if self._mode == "RGB" or self._mode == "RGBA":
                    # [..., ..., ..., 1, 3] or [..., ..., ..., 1, 4]
                    axes = Axes(["X", "Y", "Z", "T", "C"])
                else:
                    # The image is single-channel with 1 value in Temporal dimension
                    # instead of channel. So we map T to be channel.
                    # [..., ..., ..., 1]
                    axes = Axes(["X", "Y", "Z", "C"])
            else:
                # The time dimension does correspond to time
                axes = Axes(["X", "Y", "Z", "T"])
        elif dims_number < 4:
            # Only spatial dimensions
            if self._mode == "RGB" or self._mode == "RGBA":
                axes = Axes(["X", "Y", "Z", "C"])
            else:
                axes = Axes(["X", "Y", "Z"])
        else:
            # Has more dimensions that belong to spatial-temporal unknown attributes
            # TODO: investigate sample images of this format.
            if self._mode == "RGB" or self._mode == "RGBA":
                axes = Axes(["X", "Y", "Z", "C"])
            else:
                axes = Axes(["X", "Y", "Z"])

        self._logger.debug(f"Reader axes: {axes}")
        return axes

    @property
    def channels(self) -> Sequence[str]:
        if self.webp_format is WebpInputFormat.WEBP_RGB:
            self._logger.debug(f"Webp format: {WebpInputFormat.WEBP_RGB}")
            return "RED", "GREEN", "BLUE"
        elif self.webp_format is WebpInputFormat.WEBP_RGBA:
            self._logger.debug(f"Webp format: {WebpInputFormat.WEBP_RGBA}")
            return "RED", "GREEN", "BLUE", "ALPHA"
        else:
            self._logger.debug(
                f"Webp format is not: {WebpInputFormat.WEBP_RGB} / {WebpInputFormat.WEBP_RGBA}"
            )
        color_map = {
            "R": "RED",
            "G": "GREEN",
            "B": "BLUE",
            "A": "ALPHA",
        }
        # Use list comprehension to convert the short form to full form
        rgb_full = [color_map[color] for color in self._mode]
        return rgb_full

    @property
    def level_count(self) -> int:
        level_count = 1
        self._logger.debug(f"Level count: {level_count}")
        return level_count

    def level_dtype(self, level: int = 0) -> np.dtype:
        header_dict = self.nifti1_hdr_2_dict()

        dtype = get_dtype_from_code(header_dict["datatype"])
        if dtype == np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")]):
            dtype = np.uint8
        # TODO: Compare with the dtype of fields

        self._logger.debug(f"Level {level} dtype: {dtype}")
        return dtype

    def level_shape(self, level: int = 0) -> Tuple[int, ...]:
        if level != 0:
            return ()

        original_shape = self._nib_image.shape
        if (fields := self._nib_image.dataobj.dtype.fields) is not None:
            if len(fields) == 3:
                # RGB convert the shape from to stack 3 channels
                l_shape = (*original_shape, 3)
            elif len(fields) == 4:
                # RGBA
                l_shape = (*original_shape, 4)
            else:
                # Grayscale
                l_shape = original_shape
        else:
            l_shape = original_shape
        self._logger.debug(f"Level {level} shape: {l_shape}")
        return l_shape

    @property
    def webp_format(self) -> WebpInputFormat:
        self._logger.debug(f"Channel Mode: {self._mode}")
        if self._mode == "RGB":
            return WebpInputFormat.WEBP_RGB
        elif self._mode == "RGBA":
            return WebpInputFormat.WEBP_RGBA
        return WebpInputFormat.WEBP_NONE

    def level_image(
        self, level: int = 0, tile: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:

        unscaled_img = self._nib_image.dataobj.get_unscaled()
        self._metadata["original_mode"] = self._mode
        raw_data_contiguous = np.ascontiguousarray(unscaled_img)
        numerical_data = np.frombuffer(raw_data_contiguous, dtype=self.level_dtype())
        # Account endianness
        numerical_data = numerical_data.view(
            numerical_data.dtype.newbyteorder(self._nib_image.header.endianness)
        )
        numerical_data = numerical_data.reshape(self.level_shape())

        # Bug! data might have slope and inter and header not contain them.

        if tile is None:
            return numerical_data
        else:
            return numerical_data[tile]

    def level_metadata(self, level: int) -> Dict[str, Any]:
        # Common with group metadata since there are no multiple levels
        writer_kwargs = dict(metadata=self._metadata)
        return {"json_write_kwargs": json.dumps(writer_kwargs)}

    @property
    def original_metadata(self) -> Any:
        self._logger.debug(f"Original Image metadata: {self._metadata}")
        return self._metadata

    def optimal_reader(
        self, level: int, max_workers: int = 0
    ) -> Optional[Iterator[Tuple[Tuple[slice, ...], NDArray[Any]]]]:
        # if self.chunk_size is None:
        #     raise ValueError("chunk_size must be set for chunked reading.")
        #
        # array = self._nib_image.get_fdata()
        # array = self._nib_image.get_fdata()
        # total_slices = array.shape[-1]
        # for i in range(0, total_slices, self.chunk_size):
        #     chunk = array[..., i : i + self.chunk_size]
        #     yield chunk
        return None

    def nifti1_hdr_2_dict(self) -> Dict[str, Any]:
        structured_header_arr = self._nib_image.header.structarr
        return {
            field: structured_header_arr[field]
            for field in structured_header_arr.dtype.names
        }

    @staticmethod
    def _serialize_header(header_dict: Mapping[str, Any]) -> Dict[str, Any]:
        serialized_header = {
            k: (
                base64.b64encode(v.tolist()).decode("utf-8")
                if isinstance(v, np.ndarray) and isinstance(v.tolist(), bytes)
                else v.tolist() if isinstance(v, np.ndarray) else v
            )
            for k, v in header_dict.items()
        }
        return serialized_header


class NiftiWriter:
    def __init__(self, output_path: str, logger: logging.Logger):
        self._logger = logger
        self._output_path = output_path
        self._group_metadata: Dict[str, Any] = {}
        self._nifti1header = partial(nib.Nifti1Header)
        self._original_mode = None
        self._writer = partial(nib.Nifti1Image)

    def __enter__(self) -> NiftiWriter:
        return self

    def compute_level_metadata(
        self,
        baseline: bool,
        num_levels: int,
        image_dtype: np.dtype,
        group_metadata: Mapping[str, Any],
        array_metadata: Mapping[str, Any],
        **writer_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:

        writer_metadata: Dict[str, Any] = {}
        self._original_mode = group_metadata.get("original_mode", "RGB")
        writer_metadata["mode"] = self._original_mode
        self._logger.debug(f"Writer metadata: {writer_metadata}")
        return writer_metadata

    def write_group_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._group_metadata = json.loads(metadata["json_write_kwargs"])

    def _structured_dtype(self) -> Optional[np.dtype]:
        if self._original_mode == "RGB":
            return np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
        elif self._original_mode == "RGBA":
            return np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1"), ("A", "u1")])
        else:
            return None

    def write_level_image(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> None:
        header = self._nifti1header(
            binaryblock=base64.b64decode(self._group_metadata["binaryblock"])
        )
        contiguous_image = np.ascontiguousarray(image)
        structured_arr = contiguous_image.view(
            dtype=self._structured_dtype() if self._structured_dtype() else image.dtype
        )
        if len(image.shape) > 3:
            # If temporal is 1 and extra dim for channels RGB/RGBA
            if image.shape[3] == 1 and (image.shape[4] == 3 or 4):
                structured_arr = structured_arr.reshape(*image.shape[:4])

        nib_image = self._writer(
            structured_arr, header=header, affine=header.get_best_affine()
        )

        nib_image.header.set_slope_inter(
            self._group_metadata["slope"], self._group_metadata["inter"]
        )
        nib.save(nib_image, self._output_path)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class NiftiConverter(ImageConverterMixin[NiftiReader, NiftiWriter]):
    """Converter of Tiff-supported images to TileDB Groups of Arrays"""

    # https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
    _ImageReaderType = NiftiReader
    _ImageWriterType = NiftiWriter
