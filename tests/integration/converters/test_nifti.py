import nibabel as nib
import numpy as np
import pytest

import tiledb
from tests import get_path
from tiledb.bioimg.converters.nifti import NiftiConverter


def compare_nifti_images(file1, file2):
    img1 = nib.load(file1)
    img2 = nib.load(file2)

    # Compare the headers (metadata)
    if img1.header != img2.header:
        return False

    # Compare the affine matrices (spatial information)
    if not np.array_equal(img1.affine, img2.affine):
        return False

    # Compare the image data (voxel data)
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()
    if not np.array_equal(data1, data2):
        return False
    return True


@pytest.mark.parametrize(
    "filename", ["nifti/example4d.nii", "nifti/functional.nii", "nifti/standard.nii"]
)
@pytest.mark.parametrize("preserve_axes", [False, True])
@pytest.mark.parametrize("chunked", [False])
@pytest.mark.parametrize(
    "compressor, lossless",
    [
        (tiledb.ZstdFilter(level=0), True),
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
    compare_nifti_images(input_path, output_path)
