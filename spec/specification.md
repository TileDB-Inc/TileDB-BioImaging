# Biomedical Imaging Unified Data Model
### <span style="color:green">VERSION:</span>  **FMT_VERSION = 2**

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
``` 
{
'axes': 'YXC', 
'channels': '[]', 
'dataset_type': 'BIOIMG', 
'fmt_version': 2, 
'json_tiffwriter_kwargs': 
    '{
      "bigtiff": false, 
      "byteorder": "<", 
      "append": true, 
      "imagej": false, 
      "ome": false
    }', 
'levels': 
    '
      [
        {
          "level": 0, 
          "name": "l_0.tdb", 
          "axes": "CYX", 
          "shape": [3, 53760, 183808]
        }, 

        {
          "level": 1, 
          "name": "l_1.tdb", 
          "axes": "CYX", 
          "shape": [3, 26880, 91904]
        }, 

        {
          "level": 2, 
          "name": "l_2.tdb", 
          "axes": "CYX", 
          "shape": [3, 13440, 45952]
        }, 

        ... ,

'metadata': 
  '
    {
      "channels": 
        {
          "intensity": 
            [
              {
                "id": "0", 
                "name": "Channel 0", 
                "color": {"red": 255, "green": 0, "blue": 0, "alpha": 255}, 
                "min": 0.0, 
                "max": 255.0
              }, 

              {
                "id": "1", 
                "name": "Channel 1", 
                "color": {"red": 0, "green": 255, "blue": 0, "alpha": 255}, 
                "min": 0.0, 
                "max": 255.0
              }, 

              {
                "id": "2", 
                "name": "Channel 2", 
                "color": {"red": 0, "green": 0, "blue": 255, "alpha": 255}, 
                "min": 0.0, 
                "max": 255.0
              }
            ]
        }
      ,
      "axes": 
        [
          {"originalAxes": ["Y", "X", "C"], "originalShape": [53760, 183808, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 53760, 183808], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}}, 
          {"originalAxes": ["Y", "X", "C"], "originalShape": [26880, 91904, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 26880, 91904], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}},
          {"originalAxes": ["Y", "X", "C"], "originalShape": [13440, 45952, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 13440, 45952], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}},
          {"originalAxes": ["Y", "X", "C"], "originalShape": [6720, 22976, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 6720, 22976], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}},
          {"originalAxes": ["Y", "X", "C"], "originalShape": [3360, 11488, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 3360, 11488], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}},
          {"originalAxes": ["Y", "X", "C"], "originalShape": [1680, 5744, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 1680, 5744], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}},
          {"originalAxes": ["Y", "X", "C"], "originalShape": [840, 2872, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 840, 2872], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}},
          {"originalAxes": ["Y", "X", "C"], "originalShape": [420, 1436, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 420, 1436], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}},
          {"originalAxes": ["Y", "X", "C"], "originalShape": [210, 718, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 210, 718], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}},
          {"originalAxes": ["Y", "X", "C"], "originalShape": [105, 359, 3], "storedAxes": ["C", "Y", "X"], "storedShape": [3, 105, 359], "axesMapping": {"Y": ["Y"], "X": ["X"], "C": ["C"]}}
        ]
    }
  ', 
'original_metadata': 
  '
    {
      "philips_metadata": "<?xml version=\\"1.0\\" encoding=\\"UTF-8\\" ?>\\n<DataObject ObjectType=\\"DPUfsImport\\">\\n\\t<Attribute Name=\\"DICOM_ACQUISITION_DATETIME\\"......
    }
  '
'pixel_depth':
  '
    {
      "0": 1,
      "1": 1,
      "2": 1,
      "3": 1,
      "4": 1,
      "5": 1,
      "6": 1,
      "7": 1,
      "8": 1,
      "9": 1
    }
  ',
'pkg_version': '0.2.4.dev33+dirty'
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
    e.g.
    ```json
    {
        "subifds": 9,
        "metadata": {
        "axes": "YXS"
        },
        "extratags": [],
        "photometric": {
        "py/reduce": [
          {
          "py/type": "tifffile.tifffile.PHOTOMETRIC"
          },
          {
          "py/tuple": [
            6
          ]
          }
        ]
        },
        "planarconfig": {
        "py/reduce": [
          {
          "py/type": "tifffile.tifffile.PLANARCONFIG"
          },
          {
          "py/tuple": [
            1
          ]
          }
        ]
        },
        "extrasamples": {
        "py/tuple": []
        },
        "rowsperstrip": 0,
        "bitspersample": 8,
        "compression": {
        "py/reduce": [
          {
          "py/type": "tifffile.tifffile.COMPRESSION"
          },
          {
          "py/tuple": [
            7
          ]
          }
        ]
        },
        "predictor": 1,
        "subsampling": {
        "py/tuple": [
          2,
          2
        ]
        },
        "jpegtables": {
        "py/b64": "/9j/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwk...=="
        },
        "colormap": null,
        "subfiletype": 0,
        "software": "Philips DP v1.0",
        "tile": {
        "py/tuple": [
          512,
          512
        ]
        },
        "datetime": null,
        "resolution": {
        "py/tuple": [
          1,
          1
        ]
        },
        "resolutionunit": 2
    }
    ```
  - `Zarr`
      
      Stores the metadata that are inside the `.zattr` files and multiscale metadata as described in [NGFF metadata description](https://ngff.openmicroscopy.org/latest/#metadata). 
        
