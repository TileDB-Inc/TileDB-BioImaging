import numpy as np
import openslide as osd
import tiledb

from .base import ImageConverter


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
