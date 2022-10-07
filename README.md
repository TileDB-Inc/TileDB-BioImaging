<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

[![TileDB-BioImaging CI](https://github.com/TileDB-Inc/TileDB-BioImaging/actions/workflows/ci.yml/badge.svg)](https://github.com/TileDB-Inc/TileDB-BioImaging/actions/workflows/ci.yml)
![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ktsitsi/32d48185733a4e7375e80e3e35fab452/raw/gist_bioimg.json)

# TileDB-BioImaging

## General Information
This repo contains python code to work with TileDB arrays and biomedical data.

## Installation 
``
pip install .[full]
``

## Features
- Convert data from standard biomedical image formats to group of TileDB arrays

``from tiledbimg.util.convert_ome_zarr import OMEZarrConverter``

``cnv = OMEZarrConverter()``

``cnv.convert_image("data.zarr", "data.tiledb")``

## Project Status
Project is: _in progress_
