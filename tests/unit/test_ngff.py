import pytest
from tifffile import tifffile

from tests import get_path
from tiledb.bioimg.metadata import NGFFPlate, NGFFWell


@pytest.mark.parametrize(
    "filename, expected",
    [
        [
            "one-screen-one-plate-four-wells.ome.xml",
            {
                "plates": 1,
                "acquisition": {"Plate:1": 2},
                "wells": {"Plate:1": {(1, 1): 2, (1, 2): 2, (2, 1): 5, (2, 2): 2}},
            },
        ],
        [
            "two-screens-two-plates-four-wells.ome.xml",
            {
                "plates": 2,
                "acquisition": {"Plate:1": 2, "Plate:2": 1},
                "wells": {
                    "Plate:1": {(1, 1): 2, (1, 2): 2, (2, 1): 5, (2, 2): 2},
                    "Plate:2": {(1, 1): 2, (1, 2): 2, (2, 1): 5, (2, 2): 2},
                },
            },
        ],
        [
            "hcs.ome.xml",
            {
                "plates": 1,
                "acquisition": {"Plate:1": 0},
                "wells": {"Plate:1": {(0, 0): 1}},
            },
        ],
    ],
)
def test_plate_ome_to_ngff(filename, expected):
    input_path = get_path(f"ome-metadata/{filename}")
    with open(input_path) as f:
        omexml = f.read()

    plates = NGFFPlate.from_ome_tiff(tifffile.xml2dict(omexml))
    wells = NGFFWell.from_ome_tiff(tifffile.xml2dict(omexml))

    assert len(plates) == expected.get("plates")

    for key, plate in plates.items():
        assert len(plate.wells) == len(expected.get("wells").get(key))
        assert (
            expected.get("acquisition").get(key) == 0 and plate.acquisitions is None
        ) or len(plate.acquisitions) == expected.get("acquisition").get(key)
        for well in plate.wells:
            assert expected.get("wells").get(key).get(
                (well.rowIndex, well.columnIndex)
            ) == len(wells.get(key).get((well.rowIndex, well.columnIndex)).images)
