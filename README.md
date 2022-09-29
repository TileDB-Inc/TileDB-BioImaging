# TileDB-BioImaging

## General Information
This repo contains python code to work with TileDB arrays and biomedical data.

## Installation 
``
pip install tiledb-bioimaging
``

## Technologies Used
- Python - version 3.10

## Features
- Convert data from standard biomedical image formats to group of TileDB arrays

``from tiledbimg.util.convert_ome_zarr import OMEZarrConverter``

``cnv = OMEZarrConverter()``

``cnv.convert_image("data.zarr", "data.tiledb")``

## Project Status
Project is: _in progress_











