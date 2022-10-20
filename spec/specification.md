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

 - Images with multiple levels of resolution. 
 - Optionally associated labels.   
  
 Note that the number of dimensions is variable between 2 and 5 and that axis names are arbitrary., 
 metadata for details. For this example we assume an image with 5 dimensions and axes called t,c,z,y,x.


## On-disk Implementation with TileDB

In TileDB, we will implement the above as follows:

* ND Dense
* The arbitrary key-value metadata will be modeled as **TileDB group metadata** (see the TODOs section at the end).

A potential two image Biomedical Imaging dataset, composed of one or more TileDB array groups, would look as follows:

```
.                               # Root folder, potentially in S3.
    image_1.tiledb              # One image (id=1) converted to a TileDB Group.
    |-- __group                 # TileDB group directory
    |-- __meta                  # Group Metadata
    |-- l_0.tdb                 # Layer 0 modeled as a 2D-5D dense TileDB Array with integer dimensions and uint8 attribute.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316115_1661541316115_bd1ea43c738344fa8ce2357091a2e558_14
    |   `-- __schema
    |-- l_1.tdb                 # Layer 1 modeled as a 2D-5D dense TileDB Array with integer dimensions and uint8 attribute.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316135_1661541316135_e1fadcee168f4dc585eaf39d4ea42c33_14
    |   `-- __schema
    `-- l_2.tdb                 # Layer 2 modeled as a 2D-5D dense TileDB Array with integer dimensions and uint8 attribute.
        |-- __commits
        |-- __fragments
        |   `-- __1661541316146_1661541316146_bb390adff7c5474e8024b76b0478abc7_14
        `-- __schema
    |-- labels                      # Labels converted toa TileDB Group.
        |-- __group                 # TileDB label group directory.
        |-- __meta                  # LabelGroup Metadata.
        |-- l_0.tdb                 # Layer 0 modeled as a 2D-5D dense TileDB Array with integer dimensions and uint8 attribute.
        |   |-- __commits
        |   |-- __fragments
        |   |   `-- __1661541316115_1661541316115_bd1ea43c738344fa8ce2357091a2e558_14
        |   `-- __schema
        
    image_2.tiledb              # One image (id=2) converted to a TileDB Group.
    |-- __group                 # TileDB group directory
    |-- __meta                  # Group Metadata
    |-- l_0.tdb                 # Layer 0 modeled as a 2D-5D dense TileDB Array with integer dimensions and uint8 attribute.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316115_1661541316115_bd1ea43c738344fa8ce2357091a2e558_14
    |   `-- __schema
    |-- l_1.tdb                 # Layer 1 modeled as a 2D-5D dense TileDB Array with integer dimensions and uint8 attribute.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316135_1661541316135_e1fadcee168f4dc585eaf39d4ea42c33_14
    |   `-- __schema
```


