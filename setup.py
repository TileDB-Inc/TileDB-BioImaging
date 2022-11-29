import setuptools

zarr = ["ome-zarr"]
openslide = ["openslide-python"]
tiff = ["tifffile", "imagecodecs"]
full = sorted({*zarr, *openslide, *tiff})

setuptools.setup(
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb/bioimg/version.py",
    },
    install_requires=["edit_operation", "tiledb"],
    extras_require={
        "zarr": zarr,
        "openslide": openslide,
        "tiff": tiff,
        "full": full,
    },
)
