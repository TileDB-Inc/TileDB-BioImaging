<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

[![TileDB-BioImaging CI](https://github.com/TileDB-Inc/TileDB-BioImaging/actions/workflows/ci.yml/badge.svg)](https://github.com/TileDB-Inc/TileDB-BioImaging/actions/workflows/ci.yml)
![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ktsitsi/32d48185733a4e7375e80e3e35fab452/raw/gist_bioimg.json)

# TileDB-BioImaging

Python package for:
- converting images stored in popular Biomedical Imaging formats to groups of TileDB arrays (& vice versa)
- exposing an expressive and efficient API (powered by TileDB) for querying such data.

## Features

### Ingestion to TileDB Groups of Arrays
    - OME-Zarr
    - OME-Tiff
    - Open-Slide

### Export from TileDB-Bioimaging Arrays to:
    - OME-Zarr
    - OME-Tiff

### Visualization Options

- [TileDB Cloud](https://cloud.tiledb.com) includes a built-in, pyramidal multi-resolution viewer: log in to TileDB Cloud to see an example image preview [here](https://cloud.tiledb.com/biomedical-imaging/TileDB-Inc/dbb7dfcc-28b3-40e5-916f-6509a666d950/preview)
- Napari: https://github.com/TileDB-Inc/napari-tiledb-bioimg

## Quick Installation

- From PyPI:

      pip install 'tiledb-bioimg[full]'

- From source:

      git clone https://github.com/TileDB-Inc/TileDB-BioImaging.git
      cd TileDB-BioImaging

      pip install -e '.[full]'

## Windows Installation

After installing `Openslide` you should make sure that you create a link between your installation path and
the following default path `C:\openslide-win64\ `.

```cmd
mklink /D C:\openslide-win64\ [your-installation-path]\openslide-win64-20221217\
```

You can install the latest versions of `Openslide` for windows using the pre-built packages
found in the project's github page:
`https://github.com/openslide/openslide-bin/releases`

or in their website:
`https://openslide.org/download/`


## Examples
How to convert imaging data from standard biomedical formats to group of TileDB arrays.

### OME-Zarr to TileDB Group of Arrays
```python
from tiledb.bioimg.converters.ome_zarr import OMEZarrConverter
OMEZarrConverter.to_tiledb("path_to_ome_zarr_image", "tiledb_array_group_path")
```

### OME-Tiff to TileDB Group of Arrays
```python
from tiledb.bioimg.converters.ome_tiff import OMETiffConverter
OMETiffConverter.to_tiledb("path_to_ome_tiff_image", "tiledb_array_group_path")
```

### Open Slide to TileDB Group of Arrays
```python
from tiledb.bioimg.converters.openslide import OpenSlideConverter
OpenSlideConverter.to_tiledb("path_to_open_slide_image", "tiledb_array_group_path")
```

## Documentation
`API Documentation` is auto-generated. Following the instructions below:

```shell
quartodoc build && quarto preview
```
