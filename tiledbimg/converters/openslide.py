import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import openslide as osd
import tiledb

from .base import ImageConverter

DEBUG = False


class OpenSlideConverter(ImageConverter):
    def convert_image(
        self, input_path: str, output_group_path: str, level_min: int = 0
    ) -> None:
        img = osd.OpenSlide(input_path)

        tiledb.group_create(output_group_path)

        # Build image arrays
        for level in range(img.level_count)[level_min:]:
            dims = img.level_dimensions[level]

            output_img_path = self.output_level_path(output_group_path, level)

            print(f"img_path: {input_path} -- output_group_path: {output_group_path}")

            schema = self.create_schema(dims)
            tiledb.Array.create(output_img_path, schema)

            slide_data = img.read_region((0, 0), level, dims).convert("RGB")
            data = np.array(slide_data).swapaxes(0, 1)
            newdata = data.view(
                dtype=np.dtype([("", "uint8"), ("", "uint8"), ("", "uint8")])
            )
            with tiledb.open(output_img_path, "w") as A:
                A[:] = newdata

        # Write group metadata
        with tiledb.Group(output_group_path, "w") as G:
            G.meta["original_filename"] = input_path
            G.meta["level_downsamples"] = img.level_downsamples


def convert_image(
    input_path: str, output_group_path: str, doit: bool = True, level_min: int = 0
) -> None:

    """
    Convert an OpenSlide-supported image to a TileDB Group of Arrays, one
    per level.

    Usage: convert_image(img_uri, output_uri, doit=True)
    """

    converter = OpenSlideConverter()
    converter.convert_image(input_path, output_group_path)


def convert_all(path: str, output_path: str, level_min: int = 0) -> None:

    """
    Batch convert a group of .svs files in `path` to TileDB image array groups
    in `output_path`.
    """

    paths = glob.glob(f"{path}/*.svs")

    print(f"found {len(paths)} .svs files in '{path}'")

    with ProcessPoolExecutor(max_workers=8) as TP:
        for p in paths:
            basename = os.path.basename(p)
            filename = os.path.split(basename)[-1]
            imagename = os.path.splitext(filename)[0]

            print(f"input file name: {filename}")
            print(f"input exp name: {imagename}")

            group_path = os.path.join(output_path, imagename)

            print(f"output name: {group_path}")

            if not DEBUG:
                TP.submit(convert_image, p, group_path, doit=True, level_min=level_min)
            else:
                # debugging
                convert_image(p, group_path, doit=True, level_min=level_min)


# %%
if __name__ == "__main__":
    output_path = "/staging/conv1"

    convert_all("/staging/orig", output_path, level_min=1)
