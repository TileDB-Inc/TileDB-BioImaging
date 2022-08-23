
import tiledb 
import numpy as np
import os
import openslide as osld
import pytest
import tifffile
from open_slide import TileDBOpenSlide, LevelInfo, SlideInfo


ometiff_uri = "../CMU-1-Small-Region.ome.tiff"
svstiff_uri = "../CMU-1-Small-Region.svs.tiff"
g_uri = "../CMU-1-Small-Region.tiledb"


def test_ome_tiff():

   ometiff_img = tifffile.TiffFile(ometiff_uri)
   os_img = osld.open_slide(svstiff_uri)
   t = TileDBOpenSlide.from_group_uri(g_uri)

#ToDo: We need to find better test data. This data has already been downsampled without preserving the original levels.

#    assert(
#         t.level_count == 3
#     )

#    assert(
#         t.dimensions == (2220, 2967)
#     )
#    assert(
#         t.level_dimensions == ((2220, 2967), (387, 463), (1280, 431))
#     )
#    assert(
#         t.level_downsamples == None
#     )

   assert(
        t.level_info == [LevelInfo(level=0, shape=(2220, 2967)), LevelInfo(level=1, shape=(387, 463)), LevelInfo(level=2, shape=(1280, 431))]
    )




