import sys
#sys.path.append("..")
import os
import math
import numpy as np
import multiprocessing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from multiprocessing import Pool
import align_ci_helpers as align
#from utils import utils
import subprocess
import time

### TODO -- comment

### OUTPUT FORMAT -- extentname_REGION_...
### assume extent has already been reprojected!!!
def subdivide_extent(extent_src, extent_name, hv_split, overlap_width, overlap_slop, input_epsg):
    ### divide extent into subregions...
    ### idea -- draw lines, then move out to avoid slicing raster pixels?
    ### but first, directory setup
    ### set up new directory to store subdivided files...
    os.system('rm -rf ' + extent_src + "subregions")
    os.system("mkdir " + extent_src + "subregions")
    ### now if the case is trivial we can leave without doing work
    if hv_split is None or (hv_split[0] == 1 and hv_split[1] == 1):
        os.system('cp ' + extent_src + extent_name + '.shp ' + extent_src + "subregions/" + extent_name + '_' + input_epsg + '_0_0.shp')
        os.system('cp ' + extent_src + extent_name + '.shx ' + extent_src + "subregions/" + extent_name + '_' + input_epsg + '_0_0.shx')
        os.system('cp ' + extent_src + extent_name + '.prj ' + extent_src + "subregions/" + extent_name + '_' + input_epsg + '_0_0.prj')
        return
    print("subdivide - loading extent -", end="", flush=True)
    ### we are going to assume that the we have already fixed projection issues with shapefile
    ### need to open the base layer to get information about the bounding box
    extent = ogr.Open(extent_src + extent_name + ".shp")
    layer = extent.GetLayer()
    print("-> done")
    for feature in layer:
        geom = feature.GetGeometryRef()
    envelope = geom.GetEnvelope()
    print("subdivide - bounding box:", envelope)
    ### compute where the borders of ideal bounding boxes would be
    horizontalcuts = [envelope[0] + (((envelope[1] - envelope[0])/hv_split[0]) * ii) for ii in range(hv_split[0] + 1)]
    verticalcuts = [envelope[2] + (((envelope[3] - envelope[2]) / hv_split[1]) * ii) for ii in range(hv_split[1] + 1)]
    print("subdivide - horizontal cuts:", horizontalcuts)
    print("subdivide - vertical cuts:", verticalcuts)
    ### need to come up with bounding boxes and clip to them with ogr2ogr -clipsrc
    ### come up with bounding boxes
    ### here we iterate, define bounding boxes, and add overlap/slop for safety
    slice_grid = []
    mini_slice_grid = []
    for i in range(hv_split[0]):
        slice_grid.append([])
        mini_slice_grid.append([])
        for j in range(hv_split[1]):
            ### need to append string with bounding coords
            mini_slice_grid[i].append(str(horizontalcuts[i]-overlap_slop) + " " +
                                   str(verticalcuts[j]-overlap_slop) + " " +
                                   str(horizontalcuts[i+1]+overlap_slop+overlap_width) + " " +
                                   str(verticalcuts[j+1]+overlap_slop+overlap_width))

    ### get driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ### iterate over bounding boxes and make a new file for the extent sliced by each bbx -- some may have 0 area (cali)
    for i in range(len(mini_slice_grid)):
        for j in range(len(mini_slice_grid[i])):
            ### create the data source
            print("subdivide - created subregion bounding box", i, j)
            os.system('ogr2ogr -f "ESRI Shapefile" -clipsrc ' + mini_slice_grid[i][j] + ' ' +
                      extent_src + 'subregions/' + extent_name + '_' + str(input_epsg) + '_' + str(i) + '_' + str(j) +
                      '.shp ' + extent_src + extent_name + '.shp ')


gdal.UseExceptions()
### main portion
if len(sys.argv) > 1 and sys.argv[1] == "test_subdivide":
    subdivide_extent("../data/extent/california/", "CA_State", (2, 4), 0.2, 0.05, 3785)
