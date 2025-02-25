import setuptools

zarr = ["ome-zarr>=0.9.0"]
openslide = ["openslide-python"]
tiff = ["tifffile", "imagecodecs"]
nifti = ["nibabel"]
cloud = ["tiledb-cloud"]

full = sorted({*zarr, *openslide, *tiff, *nifti})
setuptools.setup(
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb/bioimg/version.py",
    },
    install_requires=[
        "openslide-bin",
        "pyeditdistance",
        "tiledb>=0.19",
        "tqdm",
        "scikit-image",
        "jsonpickle",
        "requires",
    ],
    extras_require={
        "zarr": zarr,
        "openslide": openslide,
        "tiff": tiff,
        "nifti": nifti,
        "cloud": cloud,
        "full": full,
    },
    entry_points={
        "bioimg.readers": [
            "tiff_reader = tiledb.bioimg.converters.ome_tiff:OMETiffReader",
            "zarr_reader = tiledb.bioimg.converters.ome_zarr:OMEZarrReader",
            "osd_reader = tiledb.bioimg.converters.openslide:OpenSlideReader",
            "png_reader = tiledb.bioimg.converters.png.PNGReader",
            "nifti_reader = tiledb.bioimg.converters.nifti:NiftiReader",
        ],
        "bioimg.writers": [
            "tiff_writer = tiledb.bioimg.converters.ome_tiff:OMETiffWriter",
            "zarr_writer = tiledb.bioimg.converters.ome_tiff:OMEZarrWriter",
            "png_writer = tiledb.bioimg.converters.png.PNGWriter",
            "nifti_writer = tiledb.bioimg.converters.nifti:NiftiWriter",
        ],
        "bioimg.converters": [
            "tiff_converter = tiledb.bioimg.converters.ome_tiff:OMETiffConverter",
            "zarr_converter = tiledb.bioimg.converters.ome_zarr:OMEZarrConverter",
            "osd_converter = tiledb.bioimg.converters.openslide:OpenSlideConverter",
            "png_converter = tiledb.bioimg.converters.png:PNGConverter",
            "nifti_converter = tiledb.bioimg.converters.nifti:NiftiConverter",
        ],
    },
)
