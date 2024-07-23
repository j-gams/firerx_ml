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
                      extent_src + 'subregions/' + extent_name + '_' + str(i) + '_' + str(j) + '.shp ' +
                      extent_src + extent_name + '.shp ')


gdal.UseExceptions()
### main portion
if len(sys.argv) > 1 and sys.argv[1] == "test_subdivide":
    subdivide_extent("../data/extent/colorado/", "Colorado_State_Boundary_4326", (4, 2), 0.2, 0.05, 4326)

"""
def raster_cleaning_operations(data_info, guiding_layer, extent_src, skip_guiding_load, extent_epsg_override=False):
    ### load unaligned raw data... setup lists
    raster = []
    raster_proj = []
    raster_ndvals = []
    raster_size = []
    raster_crs = []
    raster_nparray = []

    ### LOAD RAW GUIDING LAYER DATA
    if not skip_guiding_load:
        print("loading guiding layer: ", data_info[guiding_layer]["loc"])
        yraster = gdal.Open()

        print("loading guiding layer: " + align_to_layer["loc"])
        yraster = gdal.Open(raster_src + align_to_layer["loc"])
        yrband = yraster.GetRasterBand(1)
        yr_ndval = yrband.GetNoDataValue()
        yr_size = (yraster.RasterXSize, yraster.RasterYSize)
        yulh, yph, _, yulv, _, ypv = yraster.GetGeoTransform()
        ypv = abs(ypv)
        yr_proj = yraster.GetProjection()
        yr_crs = (yulh, yulv, yph, ypv)
        yr_nparray = yraster.ReadAsArray().transpose()
        yr_full_geotrans = yraster.GetGeoTransform()

        yr_epsg = osr.SpatialReference(wkt=yr_proj).GetAttrValue('AUTHORITY', 1)

        ### expect 4326 for WUE
        print("  - guiding layer epsg", yr_epsg)

    print("loading extent")
    extent = gdal.OpenEx(extent_src + ".shp", sibling_files=[extent_src + ".prj"])
    extent_proj = extent.GetLayer().GetSpatialRef()
    extent_epsg = extent_proj.GetAttrValue('AUTHORITY', 1)
    if extent_epsg_override is not False:
        print("overriding extent epsg from", extent_epsg, "to", extent_epsg_override)
        extent_epsg = extent_epsg_override
    print("extent epsg", extent_epsg)
    ### reproject to match baseline proj
    if not skip_extent_reproj:
        print("reprojecting and saving extent shapefile in original location")
        if rm_existing_TEST:
            os.system('rm ' + extent_src + '_' + str(yr_epsg) + '.shp')
        os.system(
            'ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:' + str(yr_epsg) + ' -s_srs EPSG:' + str(extent_epsg) + ' ' +
            extent_src + '_' + str(yr_epsg) + '.shp ' + extent_src + '.shp')
        correct_proj_extent = extent_src + "_" + str(yr_epsg) + ".shp"
        new_proj_base = extent_src"""