## Biomedical Imaging Unified Data Model

Biomedical imaging encompasses a wide range of imaging types and applications. At present we have primarily discussed microscopy imaging.

Opportunity:

* 1
* 2
* 3 

Initial focus:

* Import/export from all commonly used in-memory formats (e.g., OME-Zarr, OME-Tiff, Open Slide)
* Python API

Longer-term goals:

* Visualization 
* Multiple layer support and blending
* Overlays 
  * Polygon ROIs 
*Annotation
  * Draw region(s) of interest to label certain parts of an image
  * Overlay pre-computed ROIs and assign, validate, and adjust regions and labels


The core functions of the initial API are:

* Support storage of Biomedical imaging datasets in TileDB groups. 
* Simple access to dataset/collection properties.


## The Data Model

Based on our research and avoiding a lot of the jargon, there are the following “components” in typical Biomedical Imaging projects:

#OME-Zarr

Details on Biomedical imaging formats can be found [here](https://ngff.openmicroscopy.org/latest/#:~:text=OME%2DZarr%20is%20an%20implementation,in%20the%20corresponding%20Zarr%20groups.)

## On-disk Implementation with TileDB

In TileDB, we will implement the above as follows:

* ND Dense
* The arbitrary key-value metadata will be modeled as **TileDB group metadata** (see the TODOs section at the end).

A full Biomedical Imaging dataset, composed of one or more TileDB arrays, would look like:

Here goes the tree structure

# API Examples