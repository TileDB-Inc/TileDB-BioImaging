## Biomedical Imaging Unified Data Model

Biomedical imaging encompasses a wide range of imaging types and applications. 
The current document contains the specifications of storing bioimaging data as groups of TileDB arrays on the Cloud, 
based on the next-generation file format (NGFF) as described [here](https://ngff.openmicroscopy.org/latest/#ome-ngff).


## API
The core functions of the initial API are:

* Support storage of Biomedical imaging datasets stored as OME-Zarr or OME-Tiff files, in TileDB array groups. 
* Simple access to dataset/collection properties.


## The Data Model

Based on our research and avoiding a lot of the jargon, there are the following “components” in typical Biomedical Imaging projects:

 - Images with multiple levels of resolution (Note that the number of dimensions is variable between 2 and 5 and that axis names are arbitrary). 
 - Optionally associated labels.   
 - Multiple metadata as described [here](https://ngff.openmicroscopy.org/latest/#metadata). 
  
## On-disk Implementation with TileDB

In TileDB, we will implement the above as follows:

* Images are modeled as groups of ND Dense TileDB arrays with integer dimensions and one attribute of any of the [supported data types](https://docs.tiledb.com/main/how-to/arrays/creating-arrays/creating-attributes), i.e., one dense array per image level. 
* Labels are also modeled as groups of ND Dense TileDB arrays with integer dimensions and one attribute of any of the supported data types, i.e., one dense array per image level. 
* The arbitrary key-value metadata will be modeled as **TileDB group metadata**.

A potential two image Biomedical Imaging dataset, composed of one or more TileDB array groups, would look as follows:

```
.                               # Root folder, potentially in S3.
    image_1.tiledb              # One image (id=1) converted to a TileDB Group.
    |-- __group                 # TileDB group directory. 
    |-- __meta                  # Group Metadata, contains all group metadata in a Key-Value manner. Metadata include all kinds of group metadata needed based on NGFF.     
    |-- l_0.tdb                 # Layer 0 modeled as a 2D-5D dense TileDB Array with integer dimensions and uint8 attribute.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316115_1661541316115_bd1ea43c738344fa8ce2357091a2e558_14
    |   `-- __schema
    |-- l_1.tdb                 # Layer 1 modeled as a 2D-5D dense TileDB Array with integer dimensions and one attribute of any of the supported data types.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316135_1661541316135_e1fadcee168f4dc585eaf39d4ea42c33_14
    |   `-- __schema
    `-- l_2.tdb                 # Layer 2 modeled as a 2D-5D dense TileDB Array with integer dimensions and one attribute of any of the supported data types.
        |-- __commits
        |-- __fragments
        |   `-- __1661541316146_1661541316146_bb390adff7c5474e8024b76b0478abc7_14
        `-- __schema
    |-- labels                      # Labels converted toa TileDB Group.
        |-- __group                 # TileDB label group directory.
        |-- __meta                  # LabelGroup Metadata.
        |-- l_0.tdb                 # Layer 0 modeled as a 2D-5D dense TileDB Array with integer dimensions and one attribute of any of the supported data types.
        |   |-- __commits
        |   |-- __fragments
        |   |   `-- __1661541316115_1661541316115_bd1ea43c738344fa8ce2357091a2e558_14
        |   `-- __schema
        
    image_2.tiledb              # One image (id=2) converted to a TileDB Group.
    |-- __group                 # TileDB group directory
    |-- __meta                  # Group Metadata
    |-- l_0.tdb                 # Layer 0 modeled as a 2D-5D dense TileDB Array with integer dimensions and one attribute of any of the supported data types.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316115_1661541316115_bd1ea43c738344fa8ce2357091a2e558_14
    |   `-- __schema
    |-- l_1.tdb                 # Layer 1 modeled as a 2D-5D dense TileDB Array with integer dimensions and one attribute of any of the supported data types.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316135_1661541316135_e1fadcee168f4dc585eaf39d4ea42c33_14
    |   `-- __schema
```

## Metadata
As mentioned above, there are various kinds of metadata associated with any Biomedical Imaging dataset. 
The various TileDB group metadata files throughout the above array hierarchy may contain metadata keys as 
specified in the [NGFF metadata description](https://ngff.openmicroscopy.org/latest/#metadata), 
for discovering certain types of data, especially images. Specifically may contain the following **optional** metadata keys.

 1. [axes](https://ngff.openmicroscopy.org/latest/#axes-md)
 2. [bioformats2raw.layout](https://ngff.openmicroscopy.org/latest/#bf2raw)
 3. [coordinateTransformations](https://ngff.openmicroscopy.org/latest/#trafo-md)
 4. [multiscales](https://ngff.openmicroscopy.org/latest/#multiscale-md)
 5. [omero](https://ngff.openmicroscopy.org/latest/#omero-md)
 6. [labels](https://ngff.openmicroscopy.org/latest/#labels-md)
 7. [image-label](https://ngff.openmicroscopy.org/latest/#label-md)
 8. [plate](https://ngff.openmicroscopy.org/latest/#plate-md)
 9. [well](https://ngff.openmicroscopy.org/latest/#well-md) 