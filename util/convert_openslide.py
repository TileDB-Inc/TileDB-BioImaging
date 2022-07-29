#%%
from attr import attrs
import tiledb
import numpy as np
import tempfile
import glob, os, shutil

import openslide as osd

from concurrent.futures import ProcessPoolExecutor

def create_schema(img_shape):
    img_shape = tuple((img_shape[0], img_shape[1], 3)) # swappity
    print(img_shape)
    dims = []

    dims.append(
        tiledb.Dim(name="X", domain=(0,img_shape[0]-1), dtype=np.uint64, tile=1024)
    )
    dims.append(
            tiledb.Dim(name="Y", domain=(0,img_shape[1]-1), dtype=np.uint64, tile=1024)
    )

    filters = [tiledb.ZstdFilter(level=0)]
    attr = tiledb.Attr(
        name='rgb', dtype=[("", 'uint8'), ("", 'uint8'), ("", 'uint8')], filters=filters
    )

    schema = tiledb.ArraySchema(
        tiledb.Domain(*dims),
        attrs=[attr]
    )
    return schema

def convert_image(input_img_path, img_group_path, doit=True, level_min=0):
    """
    Convert an OpenSlide-supported image to a TileDB Group of Arrays, one
    per level.

    Usage: convert_image(img_uri, output_uri, doit=True)
    """
    img = osd.OpenSlide(input_img_path)

    tiledb.group_create(img_group_path)

    # Build image arrays
    for level in range(img.level_count)[level_min:]:
        dims = img.level_dimensions[level]

        output_img_path = os.path.join(img_group_path, f"l_{level}.tdb")

        print(f"img_path: {input_img_path} -- img_group_path: {img_group_path}")

        if doit:
            schema = create_schema(dims)
            tiledb.Array.create(output_img_path, schema)

            slide_data = img.read_region((0,0), level, dims).convert("RGB")
            data = np.array(slide_data).swapaxes(0,1)
            newdata = data.view(dtype=np.dtype([("", 'uint8'), ("", 'uint8'), ("", 'uint8')]))
            with tiledb.open(output_img_path, "w") as A:
                A[:] = newdata

    # Write group metadata
    with tiledb.Group(img_group_path, "w") as G:
        G.meta["original_filename"] = input_img_path
        G.meta["level_downsamples"] = img.level_downsamples

def convert_all(path, output_path, level_min=0):
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

            if True:
                TP.submit(
                    convert_image,
                    p,
                    group_path,
                    doit=True,
                    level_min=level_min
                )
            else:
                # debugging
                convert_image(
                    p,
                    group_path,
                    doit=True,
                    level_min=level_min
                )


# %%
if __name__ == "__main__":
    output_path = "/staging/conv1"

    convert_all("/staging/orig", output_path, level_min=1)
