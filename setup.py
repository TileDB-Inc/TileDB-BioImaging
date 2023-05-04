import setuptools
from setuptools import find_namespace_packages

zarr = ["ome-zarr"]
openslide = ["openslide-python"]
tiff = ["tifffile", "imagecodecs", "jsonpickle"]
cloud = ["tiledb-cloud"]

full = sorted({*zarr, *openslide, *tiff, *cloud})
setuptools.setup(
    packages=find_namespace_packages(exclude=["tests*", "docs*"]),
    setup_requires=["setuptools_scm"],
    use_scm_version={
        "version_scheme": "guess-next-dev",
        "local_scheme": "dirty-tag",
        "write_to": "tiledb/bioimg/version.py",
    },
    install_requires=["pyeditdistance", "tiledb>=0.19", "tqdm", "scikit-image"],
    extras_require={
        "zarr": zarr,
        "openslide": openslide,
        "tiff": tiff,
        "cloud": cloud,
        "full": full,
    },
)
