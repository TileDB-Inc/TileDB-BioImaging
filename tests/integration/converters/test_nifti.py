import nibabel as nib
import numpy as np
import pytest

import tiledb
from tests import get_path
from tiledb.bioimg.converters.nifti import NiftiConverter


def compare_nifti_images(file1, file2, scaled_test):
    img1 = nib.load(file1)
    img2 = nib.load(file2)

    # Compare the affine matrices (spatial information)
    assert np.array_equal(img1.affine, img2.affine)

    # Compare the image data (voxel data)
    data1 = np.array(img1.dataobj, dtype=img1.get_data_dtype())
    data2 = np.array(img2.dataobj, dtype=img2.get_data_dtype())

    assert np.array_equal(data1, data2)

    # Compare the image data scaled (voxel data)
    if scaled_test:
        data_sc = img1.get_fdata()
        data_sc_2 = img2.get_fdata()

        assert np.array_equal(data_sc, data_sc_2)


@pytest.mark.parametrize(
    "filename",
    [
        "nifti/example4d.nii",
        "nifti/functional.nii",
        "nifti/standard.nii",
        "nifti/visiblehuman.nii",
        "nifti/anatomical.nii",
    ],
)
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked", [False])
@pytest.mark.parametrize(
    "compressor, lossless",
    [
        (tiledb.ZstdFilter(level=0), False),
        # WEBP is not supported for Grayscale images
    ],
)
def test_nifti_converter_roundtrip(
    tmp_path, preserve_axes, chunked, compressor, lossless, filename
):
    # For lossy WEBP we cannot use random generated images as they have so much noise
    input_path = str(get_path(filename))
    tiledb_path = str(tmp_path / "to_tiledb")
    output_path = str(tmp_path / "from_tiledb.nii")

    NiftiConverter.to_tiledb(
        input_path,
        tiledb_path,
        preserve_axes=preserve_axes,
        chunked=chunked,
        compressor=compressor,
        log=False,
    )
    # Store it back to PNG
    NiftiConverter.from_tiledb(tiledb_path, output_path)
    # The dtype of this image is complex and nibabel breaks originally
    compare_nifti_images(
        input_path,
        output_path,
        scaled_test=False if filename == "nifti/visiblehuman.nii" else True,
    )
