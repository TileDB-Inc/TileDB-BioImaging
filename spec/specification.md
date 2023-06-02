# Biomedical Imaging Unified Data Model
### <span style="color:green">VERSION:</span>  **FMT_VERSION = 1**

## Description
Biomedical imaging encompasses a wide range of imaging types and applications. 
The current document contains the specifications of storing bioimaging data as groups of TileDB arrays on the Cloud, 
based on the next-generation file format (NGFF) as described [here](https://ngff.openmicroscopy.org/latest/#ome-ngff).


## API
The core functionalities of the `API` in this version are:

* Support storage of Biomedical imaging datasets stored as `OME-Zarr` or `OME-Tiff` files in TileDB array groups. 
* Simple access to dataset/collection properties.

### Storage
The **storage** in TileDB arrays is being offered by the following **converters** `API`:

* `OMETiffConverter`
* `OMEZarrConverter`
* `OpenSlideConverter`
  
All the above converters in this version support bidirectional conversion from the aforementioned storage formats to TileDB and back, except for `OpenSlideConverter`.

### Access

The access to the data has been designed to follow the `OpenSlide` API.
Our own `TileDBOpenSlide` API offers the same functionalities and properties like:

- `level_count: int` : returns the number of levels in the slide
- `level_dimensions: Sequence[Tuple[int, int]]`: A sequence of dimensions for each level
- etc...

You can have access to the API by building the `docs` folder inside the repo on your own:
```shell
cd docs
make html
```

## The Data Model

Based on our research and avoiding a lot of the jargon, there are the following “components” in typical Biomedical Imaging projects:

 - Images with multiple levels of resolution (Note that the number of dimensions is variable between 2 and 5 and that axis names are arbitrary). 
 - Optionally associated labels.   
 - Multiple metadata as described [here](https://ngff.openmicroscopy.org/latest/#metadata). 
  
## On-disk Implementation with TileDB

In TileDB, we will implement the above as follows:

* Images are modeled as groups of `ND Dense TileDB` arrays with integer dimensions and one attribute of any of the [supported data types](https://docs.tiledb.com/main/how-to/arrays/creating-arrays/creating-attributes), i.e., one dense array per image level. 
* Labels are also modeled as groups of `ND Dense TileDB` arrays with integer dimensions and one attribute of any of the supported data types, i.e., one dense array per image level. 
* The arbitrary key-value metadata will be modeled as `TileDB group metadata`.
* The `attribute name` of the arrays is `'intensity'`.

## On Disk Example
A potential two image Biomedical Imaging dataset, composed of one or more TileDB array groups, would look as follows:

```
.                               # Root folder, potentially in S3.
    image_1.tiledb              # One image (id=1) converted to a TileDB Group.
    |-- __group                 # TileDB group directory. 
    |-- __meta                  # Group Metadata, contains all group metadata in a 
                                # key-Value manner. Metadata include all kinds of 
                                # group metadata needed based on NGFF.     
    |-- l_0.tdb                 # Layer 0 modeled as a 2D-5D dense TileDB Array 
                                # with integer dimensions and uint8 or uint16 attribute.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316115_1661541316115_bd1ea43c738344fa8ce2357091a2e558_14
    |   `-- __schema
    |-- l_1.tdb                 # Layer 1 modeled as a 2D-5D dense TileDB Array 
                                # with integer dimensions and one attribute of 
                                # any of the supported data types.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316135_1661541316135_e1fadcee168f4dc585eaf39d4ea42c33_14
    |   `-- __schema
    `-- l_2.tdb                 # Layer 2 modeled as a 2D-5D dense TileDB Array 
                                # with integer dimensions and one attribute of 
                                # any of the supported data types.
        |-- __commits
        |-- __fragments
        |   `-- __1661541316146_1661541316146_bb390adff7c5474e8024b76b0478abc7_14
        `-- __schema
    |-- labels                      # Labels converted to TileDB Group.
        |-- __group                 # TileDB label group directory.
        |-- __meta                  # LabelGroup Metadata.
        |-- l_0.tdb                 # Layer 0 modeled as a 2D-5D dense TileDB Array 
                                    # with integer dimensions and one attribute of 
                                    # any of the supported data types.
        |   |-- __commits
        |   |-- __fragments
        |   |   `-- __1661541316115_1661541316115_bd1ea43c738344fa8ce2357091a2e558_14
        |   `-- __schema
    
    ...
    image_N.tiledb              # One image (id=N) converted to a TileDB Group.
    |-- __group                 # TileDB group directory
    |-- __meta                  # Group Metadata
    |-- l_0.tdb                 # Layer 0 modeled as a 2D-5D dense TileDB Array 
                                # with integer dimensions and one attribute of 
                                # any of the supported data types.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316115_1661541316115_bd1ea43c738344fa8ce2357091a2e558_14
    |   `-- __schema
    |-- l_1.tdb                 # Layer 1 modeled as a 2D-5D dense TileDB Array 
                                # with integer dimensions and one attribute of 
                                # any of the supported data types.
    |   |-- __commits
    |   |-- __fragments
    |   |   `-- __1661541316135_1661541316135_e1fadcee168f4dc585eaf39d4ea42c33_14
    |   `-- __schema
    ...
...
```

## Metadata
As mentioned above, there are various kinds of metadata associated with any Biomedical Imaging dataset. 

For the Biomedical images converted in TileDB from the aforementioned formats the metadata of the images are stored in 

- `TileDB Group Metadata`
```json 
{
    'axes': 'YXC', 
    'channels': '["RED", "GREEN", "BLUE", "ALPHA"]',
    'dataset_type': 'BIOIMG', 
    'fmt_version': 1, 
    'levels': '[
        {"level": 0, "name": "l_0.tdb", "axes": "YXC", "shape": [26340, 99599, 4]},
        {"level": 1, "name": "l_1.tdb", "axes": "YXC", "shape": [6585, 24899, 4]}, 
        {"level": 2, "name": "l_2.tdb", "axes": "YXC", "shape": [1646, 6224, 4]}, 
        {"level": 3, "name": "l_3.tdb", "axes": "YXC", "shape": [823, 3112, 4]}]', 
    'pixel_depth': 1, 
    'pkg_version': '0.2.1'
}
```
In the example above we can find in metadata some TileDB's imputed metadata like: 
  - `fmt_version`: refers to version that follows this spec file
  - `pgk_version`: refers to the specific release of `TileDB-Bioimg` the created this image
  - `dataset_type`: useful to find which group of TileDB arrays refers to a bioimaging dataset

The `TileDB Group metadata` may also contain metadata keys as specified in the [NGFF metadata description](https://ngff.openmicroscopy.org/latest/#metadata). Specifically may contain the following **optional** metadata keys.
```
 1. [axes]
 2. [bioformats2raw.layout]
 3. [coordinateTransformations]
 4. [multiscales]
 5. [omero]
 6. [labels]
 7. [image-label]
 8. [plate]
 9. [well]
```

- `TileDB Arrays Metadata:` Out of each of the resolutions of an image we store metadata coming from it depending its initial storage format. Alongside with them we store also some metadata like the following:
  - The `level` this resolution corresponds to in the `TileDB Group`
  - The `XML` metadata in the case where image follows the `OME` specification
  - The `axes` of the image e.g `YXS` etc..

And some source format specifics like:
  - `Tiff`
    ```
    photometric
    planarconfig
    extrasamples
    rowsperstrip
    bitspersample
    compression
    predictor
    subsampling
    jpegtables
    colormap
    subfiletype
    software
    tile
    datetime
    resolution
    resolutionunit
    ```
  - `Zarr`
      
      Stores the metadata that are inside the `.zattr` files and multiscale metadata as described in [NGFF metadata description](https://ngff.openmicroscopy.org/latest/#metadata). 
        