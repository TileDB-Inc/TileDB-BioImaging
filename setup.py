import setuptools

zarr = ["zarr"]
openslide = ["openslide-python"]
tiff = ["tiff"]
full = sorted({*zarr, *openslide, *tiff})

setuptools.setup(
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledbimg/version.py",
    },
    install_requires=["tiledb"],
    extras_require={
        "zarr": zarr,
        "openslide": openslide,
        "tiff": tiff,
        "full": full,
    },
)
