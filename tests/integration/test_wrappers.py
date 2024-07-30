import os

import pytest

from tests import get_path
from tiledb.bioimg import Converters, from_bioimg, to_bioimg
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
from tiledb.bioimg.converters.ome_zarr import OMEZarrConverter
from tiledb.bioimg.converters.openslide import OpenSlideConverter


@pytest.mark.parametrize(
    "converter, file_path",
    [
        (Converters.OMETIFF, "CMU-1-Small-Region-rgb.ome.tiff"),
        (Converters.OMETIFF, "CMU-1-Small-Region.ome.tiff"),
        (Converters.OMETIFF, "rand_uint16.ome.tiff"),
        (Converters.OMETIFF, "UTM2GTIF.tiff"),
        (Converters.OMEZARR, "CMU-1-Small-Region.ome.zarr"),
        (Converters.OSD, "CMU-1-Small-Region.svs"),
    ],
)
def test_from_bioimg_wrapper(tmp_path, converter, file_path):
    input_path = str(get_path(file_path))
    output_path = str(tmp_path)
    output_path_round = str(tmp_path) + "/roundtrip"
    if converter == Converters.OMETIFF:
        rfromtype = from_bioimg(input_path, output_path, converter=converter)
        rtotype = to_bioimg(output_path, output_path_round, converter=converter)
        assert rfromtype == rtotype == OMETiffConverter
    elif converter == Converters.OSD:
        rtype = from_bioimg(input_path, output_path, converter=converter)
        assert rtype == OpenSlideConverter
        with pytest.raises(NotImplementedError):
            to_bioimg(output_path, output_path_round, converter=converter)
    else:
        input_path = os.path.join(input_path, str(0))
        rfromtype = from_bioimg(input_path, output_path, converter=converter)
        rtotype = to_bioimg(output_path, output_path_round, converter=converter)
        assert rfromtype == rtotype == OMEZarrConverter
