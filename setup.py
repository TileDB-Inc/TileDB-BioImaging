import setuptools

zarr = [
    "ome-zarr",
]
openslide = ["openslide-python"]
tiff = [
    "tifffile@git+https://github.com/TileDB-Inc/tifffile.git@gsa/python-3.7",
    "imagecodecs",
]

isyntax = ["openphi"]

full = sorted({*zarr, *openslide, *tiff, *isyntax})

setuptools.setup(
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb/bioimg/version.py",
    },
    install_requires=["levenshtein", "tiledb"],
    extras_require={
        "zarr": zarr,
        "openslide": openslide,
        "isyntax": isyntax,
        "tiff": tiff,
        "full": full,
    },
)
