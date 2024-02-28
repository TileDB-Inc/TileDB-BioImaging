import pytest
from tifffile import tifffile

from tests import get_path
from tiledb.bioimg.metadata import NGFFPlate, NGFFWell


@pytest.mark.parametrize(
    "filename",
    ["one-screen-one-plate-four-wells.ome.xml"],
)
def test_plate_ome_to_ngff(filename):
    input_path = get_path(f"ome-metadata/{filename}")
    with open(input_path) as f:
        omexml = f.read()

    NGFFPlate.from_ome_tiff(tifffile.xml2dict(omexml))
    NGFFWell.from_ome_tiff(tifffile.xml2dict(omexml))

    print(input_path)
