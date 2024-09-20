import xml.etree.ElementTree as ET

import numpy as np
import pytest

import tiledb
from tiledb.bioimg.helpers import (
    get_decimal_from_rgba,
    get_pixel_depth,
    get_rgba,
    iter_color,
    merge_ned_ranges,
    remove_ome_image_metadata,
)

from .. import generate_test_case, generate_xml


def test_color_iterator():
    generator_grayscale = iter_color(np.dtype(np.uint8), 1)
    generator_rgb = iter_color(np.dtype(np.uint8), 3)
    generator_random = iter_color(np.dtype(np.uint8), 5)

    for _, color in zip(range(1), generator_grayscale):
        assert color == get_rgba(get_decimal_from_rgba(color))
        assert color == {"red": 255, "green": 255, "blue": 255, "alpha": 255}

    for idx, color in zip(range(3), generator_rgb):
        assert color == get_rgba(get_decimal_from_rgba(color))
        assert color == {
            "red": 255 if idx == 0 else 0,
            "green": 255 if idx == 1 else 0,
            "blue": 255 if idx == 2 else 0,
            "alpha": 255,
        }

    for _, color in zip(range(5), generator_random):
        assert color == get_rgba(get_decimal_from_rgba(color))

    with pytest.raises(NotImplementedError):
        generator_non_float = iter_color(np.dtype(float), 5)
        for _, color in zip(range(5), generator_non_float):
            pass


def test_get_pixel_depth():
    webp_filter = tiledb.WebpFilter()
    # Test that for some reason input_format gets a random value not supported
    webp_filter._input_format = 6
    with pytest.raises(ValueError) as err:
        get_pixel_depth(webp_filter)
        assert "Invalid WebpInputFormat" in str(err)


@pytest.mark.parametrize(
    "num_axes, num_ranges, max_value",
    [(5, 10, 100), (3, 20, 50), (4, 15, 200), (6, 25, 300)],
)
def test_validate_ingestion(num_axes, num_ranges, max_value):
    input_ranges, expected_output = generate_test_case(num_axes, num_ranges, max_value)
    assert merge_ned_ranges(input_ranges) == expected_output


@pytest.mark.parametrize("macro", [True, False])
@pytest.mark.parametrize("has_label", [True, False])
@pytest.mark.parametrize("num_images", [1, 2, 3])
@pytest.mark.parametrize("root_tag", ["OME", "InvalidRoot"])
def test_remove_ome_image_metadata(macro, has_label, num_images, root_tag):
    original_xml_string = generate_xml(
        has_macro=macro, has_label=has_label, num_images=1, root_tag=root_tag
    )

    excluded_metadata = remove_ome_image_metadata(original_xml_string)
    if root_tag == "OME":
        parsed_excluded = ET.fromstring(excluded_metadata)

        # Assert if "label" subelement is present
        assert (
            parsed_excluded.find(
                ".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image[@ID='Image:label'][@Name='label']"
            )
            is None
        )

        # Assert if "macro" subelement is present
        assert (
            parsed_excluded.find(
                ".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image[@ID='Image:macro'][@Name='macro']"
            )
            is None
        )
    else:
        assert remove_ome_image_metadata(original_xml_string) is None
