# import os
#
# os.add_dll_directory("C:/Openslide/bin")
import numpy as np
import openslide
import PIL.Image

import pytest

import tiledb
from tests import get_path, get_schema
from tiledb.bioimg.compressor_factory import (
    WebpArguments,
    ZstdArguments,
    createCompressor,
)
from tiledb.bioimg.converters.openslide import OpenSlideConverter
from tiledb.bioimg.openslide import TileDBOpenSlide

@pytest.mark.parametrize(
    "compressor",
    [
        ZstdArguments(level=0),
        WebpArguments(quality=0, lossless=False),
        WebpArguments(quality=100, lossless=False),
        WebpArguments(quality=100, lossless=True),
    ],
)
@pytest.mark.parametrize("preserve_axes", [False, True])
def test_openslide_converter(tmp_path, compressor, preserve_axes):
    svs_path = get_path("CMU-1-Small-Region.svs")
    OpenSlideConverter.to_tiledb(
        svs_path, str(tmp_path), compressor_arguments=compressor, preserve_axes=preserve_axes
    )

    assert len(tiledb.Group(str(tmp_path))) == 1
    with tiledb.open(str(tmp_path / "l_0.tdb")) as A:
        if isinstance(compressor, WebpArguments):
            assert A.schema == get_schema(
                2220, 2967, is_webp=True, compressor=createCompressor(compressor)
            )
        else:
            if not preserve_axes:
                assert A.schema == get_schema(2220, 2967)

    o = openslide.open_slide(svs_path)
    with TileDBOpenSlide.from_group_uri(str(tmp_path)) as t:
        assert t.level_count == o.level_count
        assert t.dimensions == o.dimensions
        assert t.level_dimensions == o.level_dimensions
        assert t.level_downsamples == o.level_downsamples

        region = t.read_region(level=0, location=(100, 100), size=(300, 400))
        assert isinstance(region, np.ndarray)
        assert region.ndim == 3
        assert region.dtype == np.uint8
        img = PIL.Image.fromarray(region)
        assert img.size == (300, 400)
