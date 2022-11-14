import setuptools

zarr = [
    "ome-zarr",
]
openslide = ["openslide-python"]
tiff = [
    "tifffile@git+https://github.com/TileDB-Inc/tifffile.git@gsa/python-3.7",
    "imagecodecs",
]
full = sorted({*zarr, *openslide, *tiff})

setuptools.setup(
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb/version.py",
    },
    install_requires=["levenshtein", "tiledb"],
    extras_require={
        "zarr": zarr,
        "openslide": openslide,
        "tiff": tiff,
        "full": full,
    },
)
