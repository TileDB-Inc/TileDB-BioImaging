from pathlib import Path

import random
import numpy as np
from skimage.metrics import structural_similarity

import tiledb
from tiledb.bioimg import ATTR_NAME
from tiledb.cc import WebpInputFormat
from tiledb.bioimg.helpers import merge_ned_ranges
import xml.etree.ElementTree as ET

DATA_DIR = Path(__file__).parent / "data"


def get_schema(x_size, y_size, c_size=3, compressor=tiledb.ZstdFilter(level=0)):
    dims = []
    x_tile = min(x_size, 1024)
    y_tile = min(y_size, 1024)
    # WEBP Compressor does not accept specific dtypes so for dimensions we use the default
    dim_compressor = tiledb.ZstdFilter(level=0)
    if not isinstance(compressor, tiledb.WebpFilter):
        dim_compressor = compressor
    if isinstance(compressor, tiledb.WebpFilter):
        x_size *= c_size
        x_tile *= c_size
        if compressor.input_format == WebpInputFormat.WEBP_NONE:
            if c_size == 3:
                input_format = WebpInputFormat.WEBP_RGB
            elif c_size == 4:
                input_format = WebpInputFormat.WEBP_RGBA
            else:
                assert False, f"No WebpInputFormat with pixel_depth={c_size}"
            compressor = tiledb.WebpFilter(
                input_format=input_format,
                quality=compressor.quality,
                lossless=compressor.lossless,
            )
    else:
        if c_size > 1:
            dims.append(
                tiledb.Dim(
                    "C",
                    (0, c_size - 1),
                    tile=c_size,
                    dtype=np.uint32,
                    filters=tiledb.FilterList([compressor]),
                )
            )

    dims.append(
        tiledb.Dim(
            "Y",
            (0, y_size - 1),
            tile=y_tile,
            dtype=np.uint32,
            filters=tiledb.FilterList([dim_compressor]),
        )
    )
    dims.append(
        tiledb.Dim(
            "X",
            (0, x_size - 1),
            tile=x_tile,
            dtype=np.uint32,
            filters=tiledb.FilterList([dim_compressor]),
        )
    )

    return tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        attrs=[
            tiledb.Attr(
                name=ATTR_NAME, dtype=np.uint8, filters=tiledb.FilterList([compressor])
            )
        ],
    )


def get_path(uri):
    return DATA_DIR / uri


def assert_image_similarity(im1, im2, min_threshold=0.95, channel_axis=-1, win_size=11):
    s = structural_similarity(im1, im2, channel_axis=channel_axis, win_size=win_size)
    assert s >= min_threshold, (s, min_threshold, im1.shape)


def generate_test_case(num_axes, num_ranges, max_value):
    """
    Generate a test case with a given number of axes and ranges.

    Parameters:
    num_axes (int): Number of axes.
    num_ranges (int): Number of ranges.
    max_value (int): Maximum value for range endpoints.

    Returns:
    tuple: A tuple containing the generated input and the expected output.
    """
    input_ranges = []

    for _ in range(num_ranges):
        ranges = []
        for _ in range(num_axes):
            start = random.randint(0, max_value - 1)
            end = random.randint(start, max_value)
            ranges.append((start, end))
        input_ranges.append(tuple(ranges))

    input_ranges = tuple(input_ranges)

    expected_output = merge_ned_ranges(input_ranges)

    return input_ranges, expected_output


def generate_xml(has_macro=True, has_label=True, root_tag="OME", num_images=1):
    """Generate synthetic XML strings with options to include 'macro' and 'label' images."""

    # Create the root element
    ome = ET.Element(
        root_tag,
        {
            "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "Creator": "tifffile.py 2023.7.4",
            "UUID": "urn:uuid:40348664-c1f8-11ee-a19b-58112295faaf",
            "xsi:schemaLocation": "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd",
        },
    )

    # Create an instrument element
    instrument = ET.SubElement(ome, "Instrument", ID="Instrument:95")
    objective = ET.SubElement(
        instrument, "Objective", ID="Objective:95", NominalMagnification="40.0"
    )

    # Create standard image elements
    for i in range(num_images):
        image = ET.SubElement(ome, "Image", ID=f"Image:{i}", Name=f"Image{i}")
        pixels = ET.SubElement(
            image,
            "Pixels",
            DimensionOrder="XYCZT",
            ID=f"Pixels:{i}",
            SizeC="3",
            SizeT="1",
            SizeX="86272",
            SizeY="159488",
            SizeZ="1",
            Type="uint8",
            Interleaved="true",
            PhysicalSizeX="0.2827",
            PhysicalSizeY="0.2827",
        )
        channel = ET.SubElement(
            pixels, "Channel", ID=f"Channel:{i}:0", SamplesPerPixel="3"
        )
        tiffdata = ET.SubElement(pixels, "TiffData", PlaneCount="1")

    # Conditionally add 'macro' and 'label' images
    if has_label:
        label_image = ET.SubElement(ome, "Image", ID="Image:label", Name="label")
        pixels = ET.SubElement(
            label_image,
            "Pixels",
            DimensionOrder="XYCZT",
            ID="Pixels:label",
            SizeC="3",
            SizeT="1",
            SizeX="604",
            SizeY="594",
            SizeZ="1",
            Type="uint8",
            Interleaved="true",
            PhysicalSizeX="43.0",
            PhysicalSizeY="43.0",
        )
        channel = ET.SubElement(
            pixels, "Channel", ID="Channel:label:0", SamplesPerPixel="3"
        )
        tiffdata = ET.SubElement(pixels, "TiffData", IFD="1", PlaneCount="1")

    if has_macro:
        macro_image = ET.SubElement(ome, "Image", ID="Image:macro", Name="macro")
        pixels = ET.SubElement(
            macro_image,
            "Pixels",
            DimensionOrder="XYCZT",
            ID="Pixels:macro",
            SizeC="3",
            SizeT="1",
            SizeX="604",
            SizeY="1248",
            SizeZ="1",
            Type="uint8",
            Interleaved="true",
            PhysicalSizeX="43.0",
            PhysicalSizeY="43.0",
        )
        channel = ET.SubElement(
            pixels, "Channel", ID="Channel:macro:0", SamplesPerPixel="3"
        )
        tiffdata = ET.SubElement(pixels, "TiffData", IFD="2", PlaneCount="1")

    # Convert the ElementTree to a string
    return ET.tostring(ome, encoding="unicode")
