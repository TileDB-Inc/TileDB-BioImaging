#%%
from attr import attrs
import tiledb
import numpy as np
import tempfile
import glob, os, shutil

import openslide as osd

from concurrent.futures import ProcessPoolExecutor

tpath = "/Users/inorton/work/scratch/2022/0711-jnj-path-demo/test4-small/C3L-02964-26.svs"
#%%
def create_schema(img_shape):
    img_shape = tuple((img_shape[0], img_shape[1], 3)) # swappity
    print(img_shape)
    dims = []
    for i,ext in enumerate(img_shape):
        tile = 1024 if i < 2 else 1
        dims.append(
            tiledb.Dim(domain=(0,ext-1), dtype=np.uint64, tile=tile)
        )

    filters = [tiledb.ZstdFilter(level=0)]
    #filters = None
    attr = tiledb.Attr(
        name='', dtype='uint8', filters=filters
    )

    schema = tiledb.ArraySchema(
        tiledb.Domain(*dims),
        attrs=[attr]
    )
    return schema

#s = create_schema(img.level_dimensions[2])
#print(s)

#%%
def convert_image(input_img_path, img_group_path, doit=True, level_min=0):
    img = osd.OpenSlide(input_img_path)

    if doit:
        tiledb.group_create(img_group_path)

    for level in range(img.level_count)[level_min:]:
        dims = img.level_dimensions[level]

        output_img_path = os.path.join(img_group_path, f"l_{level}.tdb")

        print(f"img_path: {input_img_path} -- img_group_path: {img_group_path}")

        if doit:
            schema = create_schema(dims)
            tiledb.Array.create(output_img_path, schema)

            slide_data = img.read_region((0,0), level, dims).convert("RGB")
            data = np.array(slide_data).swapaxes(0,1)
            with tiledb.open(output_img_path, "w") as A:
                A[:] = data

    if doit:
        # TODO Collect more metadata
        with tiledb.Group(img_group_path, "w") as G:
            G.meta["original_filename"] = input_img_path
            G.meta["level_downsamples"] = img.level_downsamples



img_uri = "/Users/inorton/work/scratch/2022/0711-jnj-path-demo/test4-small/C3N-02572-22.svs"
output_uri = "/Users/inorton/work/scratch/2022/0711-jnj-path-demo/test4-small/convert-test/C3N-02572-22.tdg"

if True and os.path.isdir(output_uri):
    breakpoint()
    shutil.rmtree(output_uri)

convert_image(
    img_uri,
    output_uri,
    True
)

#%%
def convert_all(path, output_path, level_min=0):
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

            #convert_image(
            #    p,
            #    group_path,
            #    doit=True,
            #    level_min=level_min
            #)

            TP.submit(
                convert_image,
                p,
                group_path,
                doit=True,
                level_min=level_min
            )

# %%
if __name__ == "__main__":
    #tmp = tempfile.mkdtemp()

    output_path = "/staging/conv1"

    convert_all("/staging/orig", output_path, level_min=1)
