from typing import Any, Dict, List

import tifffile
from tifffile import TiffFile, TiffPageSeries


def qpi_original_meta(file: TiffFile) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []

    for page in file.pages.pages:
        metadata.append({"description": page.description, "tags": page.tags})

    return metadata


def qpi_image_meta(baseline: TiffPageSeries) -> Dict[str, Any]:
    # https://downloads.openmicroscopy.org/images/Vectra-QPTIFF/perkinelmer/PKI_Image%20Format.docx
    # Read the channel information from the tiff pages
    metadata: Dict[str, Any] = {"channels": []}

    metadata["physicalSizeX"] = metadata["physicalSizeΥ"] = (
        1 / baseline.keyframe.resolution[0]
    )
    metadata["physicalSizeΧUnit"] = metadata["physicalSizeΥUnit"] = "cm"
    for idx, page in enumerate(baseline._pages):
        page_metadata = tifffile.xml2dict(page.description).get(
            "PerkinElmer-QPI-ImageDescription", {}
        )
        if page.photometric == tifffile.PHOTOMETRIC.RGB:
            pass
        else:
            metadata["channels"].append(
                {
                    "name": page_metadata.get("Name", f"Channel {idx}"),
                    "id": f"{idx}",
                    "color": {
                        name: int(value)
                        for name, value in zip(
                            ["red", "green", "blue", "alpha"],
                            page_metadata.get("Color", "255,255,255").split(",")
                            + ["255"],
                        )
                    },
                }
            )

    return metadata
