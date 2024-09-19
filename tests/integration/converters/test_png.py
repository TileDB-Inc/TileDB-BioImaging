import json

import numpy as np
import pytest
from PIL import Image, ImageChops

import tiledb
from tests import assert_image_similarity, get_path, get_schema
from tiledb.bioimg.converters import DATASET_TYPE, FMT_VERSION
from tiledb.bioimg.converters.png import PNGConverter
from tiledb.bioimg.helpers import open_bioimg
from tiledb.bioimg.openslide import TileDBOpenSlide
from tiledb.cc import WebpInputFormat


def create_synthetic_image(
    mode="RGB", width=100, height=100, filename="synthetic_image.png"
):
    """
    Creates a synthetic image with either RGB or RGBA channels and saves it as a PNG file.

    Parameters:
    - image_type: 'RGB' for 3 channels, 'RGBA' for 4 channels.
    - width: width of the image.
    - height: height of the image.
    - filename: filename to store the image as a PNG.
    """
    if mode == "RGB":
        # Create a (height, width, 3) NumPy array with random values for RGB
        data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif mode == "RGBA":
        # Create a (height, width, 4) NumPy array with random values for RGBA
        data = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
    else:
        raise ValueError("Other image type are tested with sample images.")
    # Convert NumPy array to a Pillow Image
    image = Image.fromarray(data, mode)
    # Save the image as a PNG
    image.save(filename)
    return filename


def test_png_converter(tmp_path):
    input_path = str(get_path("pngs/PNG_1_L.png"))
    output_path = str(tmp_path)

    PNGConverter.to_tiledb(input_path, output_path)

    with TileDBOpenSlide(output_path) as t:
        assert len(tiledb.Group(output_path)) == t.level_count == 1
        schemas = get_schema(1080, 1080, c_size=1)
        # Storing the images as 3-channel images in CYX format
        # the slicing below using negative indexes to extract
        # the last two elements in schema's shape.
        assert t.dimensions == schemas.shape[:-3:-1]
        for i in range(t.level_count):
            assert t.level_dimensions[i] == schemas.shape[:-3:-1]
            with open_bioimg(str(tmp_path / f"l_{i}.tdb")) as A:
                assert A.schema == schemas

        region = t.read_region(level=0, location=(100, 100), size=(300, 400))
        assert isinstance(region, np.ndarray)
        assert region.ndim == 3
        assert region.dtype == np.uint8
        img = Image.fromarray(region.squeeze())
        assert img.size == (300, 400)

        for level in range(t.level_count):
            region_data = t.read_region((0, 0), level, t.level_dimensions[level])
            level_data = t.read_level(level)
            np.testing.assert_array_equal(region_data, level_data)


@pytest.mark.parametrize("filename", ["pngs/PNG_1_L.png"])
def test_png_converter_group_metadata(tmp_path, filename):
    input_path = get_path(filename)
    tiledb_path = str(tmp_path / "to_tiledb")
    PNGConverter.to_tiledb(input_path, tiledb_path, preserve_axes=False)

    with TileDBOpenSlide(tiledb_path) as t:
        group_properties = t.properties
        assert group_properties["dataset_type"] == DATASET_TYPE
        assert group_properties["fmt_version"] == FMT_VERSION
        assert isinstance(group_properties["pkg_version"], str)
        assert group_properties["axes"] == "XY"
        assert group_properties["channels"] == json.dumps(["GRAYSCALE"])

        levels_group_meta = json.loads(group_properties["levels"])
        assert t.level_count == len(levels_group_meta)
        for level, level_meta in enumerate(levels_group_meta):
            assert level_meta["level"] == level
            assert level_meta["name"] == f"l_{level}.tdb"

            level_axes = level_meta["axes"]
            shape = level_meta["shape"]
            level_width, level_height = t.level_dimensions[level]
            assert level_axes == "YX"
            assert len(shape) == len(level_axes)
            assert shape[level_axes.index("X")] == level_width
            assert shape[level_axes.index("Y")] == level_height


def compare_png(p1: Image, p2: Image, lossless: bool = True):
    if lossless:
        diff = ImageChops.difference(p1, p2)
        assert diff.getbbox() is None
    else:
        try:
            # Default min_threshold is 0.95
            assert_image_similarity(np.array(p1), np.array(p2), channel_axis=-1)
        except AssertionError:
            try:
                # for PNGs the min_threshold for WEBP lossy is < 0.85
                assert_image_similarity(
                    np.array(p1), np.array(p2), min_threshold=0.84, channel_axis=-1
                )
            except AssertionError:
                assert False


# PIL.Image does not support chunked reads/writes
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked", [False])
@pytest.mark.parametrize(
    "compressor, lossless",
    [
        (tiledb.ZstdFilter(level=0), True),
        (tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=True), True),
        (tiledb.WebpFilter(WebpInputFormat.WEBP_NONE, lossless=True), True),
    ],
)
@pytest.mark.parametrize(
    "mode, width, height",
    [
        ("RGB", 200, 200),  # Square RGB image
        ("RGB", 150, 100),  # Uneven dimensions
        ("RGB", 50, 150),  # Tall image
    ],
)
def test_png_converter_RGB_roundtrip(
    tmp_path, preserve_axes, chunked, compressor, lossless, mode, width, height
):

    input_path = str(tmp_path / f"test_{mode.lower()}_image_{width}x{height}.png")
    # Call the function to create a synthetic image
    create_synthetic_image(mode=mode, width=width, height=height, filename=input_path)
    tiledb_path = str(tmp_path / "to_tiledb")
    output_path = str(tmp_path / "from_tiledb")
    PNGConverter.to_tiledb(
        input_path,
        tiledb_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
        compressor=compressor,
        log=False,
    )
    # Store it back to PNG
    PNGConverter.from_tiledb(tiledb_path, output_path)
    compare_png(Image.open(input_path), Image.open(output_path), lossless=lossless)


@pytest.mark.parametrize("filename", ["pngs/PNG_1_L.png"])
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked", [False])
@pytest.mark.parametrize(
    "compressor, lossless",
    [
        (tiledb.ZstdFilter(level=0), True),
        # WEBP is not supported for Grayscale images
    ],
)
def test_png_converter_L_roundtrip(
    tmp_path, preserve_axes, chunked, compressor, lossless, filename
):
    # For lossy WEBP we cannot use random generated images as they have so much noise
    input_path = str(get_path(filename))
    tiledb_path = str(tmp_path / "to_tiledb")
    output_path = str(tmp_path / "from_tiledb")

    PNGConverter.to_tiledb(
        input_path,
        tiledb_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
        compressor=compressor,
        log=False,
    )
    # Store it back to PNG
    PNGConverter.from_tiledb(tiledb_path, output_path)
    compare_png(Image.open(input_path), Image.open(output_path), lossless=lossless)


@pytest.mark.parametrize("filename", ["pngs/PNG_2_RGB.png"])
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked", [False])
@pytest.mark.parametrize(
    "compressor, lossless",
    [
        (tiledb.WebpFilter(WebpInputFormat.WEBP_RGB, lossless=False), False),
    ],
)
def test_png_converter_RGB_roundtrip_lossy(
    tmp_path, preserve_axes, chunked, compressor, lossless, filename
):
    # For lossy WEBP we cannot use random generated images as they have so much noise
    input_path = str(get_path(filename))
    tiledb_path = str(tmp_path / "to_tiledb")
    output_path = str(tmp_path / "from_tiledb")

    PNGConverter.to_tiledb(
        input_path,
        tiledb_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
        compressor=compressor,
        log=False,
    )
    # Store it back to PNG
    PNGConverter.from_tiledb(tiledb_path, output_path)
    compare_png(Image.open(input_path), Image.open(output_path), lossless=lossless)


@pytest.mark.parametrize("preserve_axes", [False])
# PIL.Image does not support chunked reads/writes
@pytest.mark.parametrize("chunked", [False])
@pytest.mark.parametrize(
    "mode, width, height",
    [
        ("RGBA", 200, 200),  # Square RGBA image
        ("RGBA", 300, 150),  # Uneven dimensions
        ("RGBA", 120, 240),  # Tall image
    ],
)
@pytest.mark.parametrize(
    "compressor, lossless",
    [
        (tiledb.ZstdFilter(level=0), True),
        (tiledb.WebpFilter(WebpInputFormat.WEBP_RGBA, lossless=True), True),
        (tiledb.WebpFilter(WebpInputFormat.WEBP_NONE, lossless=True), True),
    ],
)
def test_png_converter_RGBA_roundtrip(
    tmp_path, preserve_axes, chunked, compressor, lossless, mode, width, height
):
    input_path = str(tmp_path / f"test_{mode.lower()}_image_{width}x{height}.png")
    # Call the function to create a synthetic image
    create_synthetic_image(mode=mode, width=width, height=height, filename=input_path)
    tiledb_path = str(tmp_path / "to_tiledb")
    output_path = str(tmp_path / "from_tiledb")
    PNGConverter.to_tiledb(
        input_path,
        tiledb_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
        compressor=compressor,
        log=False,
    )
    # Store it back to PNG
    PNGConverter.from_tiledb(tiledb_path, output_path)
    compare_png(Image.open(input_path), Image.open(output_path), lossless=lossless)


@pytest.mark.parametrize("filename", ["pngs/PNG_2_RGBA.png"])
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked", [False])
@pytest.mark.parametrize(
    "compressor, lossless",
    [
        (tiledb.WebpFilter(WebpInputFormat.WEBP_RGBA, lossless=False), False),
    ],
)
def test_png_converter_RGBA_roundtrip_lossy(
    tmp_path, preserve_axes, chunked, compressor, lossless, filename
):
    # For lossy WEBP we cannot use random generated images as they have so much noise
    input_path = str(get_path(filename))
    tiledb_path = str(tmp_path / "to_tiledb")
    output_path = str(tmp_path / "from_tiledb")

    PNGConverter.to_tiledb(
        input_path,
        tiledb_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
        compressor=compressor,
        log=False,
    )
    # Store it back to PNG
    PNGConverter.from_tiledb(tiledb_path, output_path)
    compare_png(Image.open(input_path), Image.open(output_path), lossless=lossless)
