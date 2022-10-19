## Biomedical Imaging Unified Data Model

Biomedical imaging encompasses a wide range of imaging types and applications. 
The current document contains the specifications of storing bioimaging data as groups of TileDB arrays on the Cloud, 
based on the next-generation file format (NGFF) as described [here](https://ngff.openmicroscopy.org/latest/#ome-ngff).


## API
The core functions of the initial API are:

* Support storage of Biomedical imaging datasets in TileDB array groups. 
* Simple access to dataset/collection properties.


## The Data Model

Based on our research and avoiding a lot of the jargon, there are the following “components” in typical Biomedical Imaging projects:

 - Images with multiple levels of resolutions 
 - Optionally associated labels. 
  
  
 Note that the number of dimensions is variable between 2 and 5 and that axis names are arbitrary., 
 metadata for details. For this example we assume an image with 5 dimensions and axes called t,c,z,y,x.


## On-disk Implementation with TileDB

In TileDB, we will implement the above as follows:

* ND Dense
* The arbitrary key-value metadata will be modeled as **TileDB group metadata** (see the TODOs section at the end).

A full Biomedical Imaging dataset, composed of one or more TileDB arrays, would look like:

Here goes the tree structure