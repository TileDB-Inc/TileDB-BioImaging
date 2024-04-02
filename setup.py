import setuptools

zarr = ["ome-zarr"]
openslide = ["openslide-python"]
tiff = ["tifffile", "imagecodecs"]
cloud = ["tiledb-cloud"]

full = sorted({*zarr, *openslide, *tiff, *cloud})
setuptools.setup(
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb/bioimg/version.py",
    },
    install_requires=[
        "pyeditdistance",
        "tiledb>=0.19",
        "tqdm",
        "scikit-image",
        "jsonpickle",
    ],
    extras_require={
        "zarr": zarr,
        "openslide": openslide,
        "tiff": tiff,
        "cloud": cloud,
        "full": full,
    },
)
