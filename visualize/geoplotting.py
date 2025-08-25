### ground work for spatial plots for paper...
### need 'd2' environment for this not firerx
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import sys
from osgeo import gdal
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

### with everything computed here...
### general idea is to load evt, cc, evc, standage wuehat, esihat, agbhat, ...
### sampling is again going to be an issue
### the idea is to sample at a 10m grid (GCF of everything) and average...

savedflag = False


### set up files we want to load in...
ad2 = {"lf_evt": {"loc": "../visualize/geographic/geo_masked.tif", #"loc": "../data/aligned_raster/analysis_nf_areas_raster/reprojected/LANDFIRE_EVT_22_mode_4326_0_0.tif",
                         "res": 30, "data": None, "crs": None, "ndv": None, "rs": None}, 
       "lf_cc": {"loc": "../data/aligned_raster/analysis_nf_areas_raster/reprojected/LANDFIRE_CC_22_mean_4326_0_0.tif",
                         "res": 30, "data": None, "crs": None, "ndv": None, "rs": None}, 
       "lf_evc": {"loc": "../data/aligned_raster/analysis_nf_areas_raster/reprojected/LANDFIRE_EVC_22_mean_4326_0_0.tif",
                         "res": 30, "data": None, "crs": None, "ndv": None, "rs": None}, 
       "standage": {"loc": "../data/aligned_raster/analysis_nf_areas_raster/reprojected/STANDAGE_06_mean_4326_0_0.tif",
                           "res": 1000, "data": None, "crs": None, "ndv": None, "rs": None}, 
       "yhat_wue": {"loc": "../visualize/geographic/c2_late_a_pyramid_predictions_1_wue.tif",
                           "res": 70, "data": None, "crs": None, "ndv": None, "rs": None},
       "yhat_esi": {"loc": "../visualize/geographic/c2_late_a_pyramid_predictions_1_esi.tif",
                           "res": 70, "data": None, "crs": None, "ndv": None, "rs": None}, 
       "yhat_agb": {"loc": "../visualize/geographic/c2_late_a_pyramid_predictions_1_agb.tif",
                            "res": 1000, "data": None, "crs": None, "ndv": None, "rs": None}}

### load in min/max scaling values 
combined_min = np.genfromtxt("../data/ml_sets/pyramid_v10/norm_layer_mins_combined.csv", delimiter=',')
combined_max = np.genfromtxt( "../data/ml_sets/pyramid_v10/norm_layer_maxs_combined.csv", delimiter=',')
print(len(combined_min), len(combined_max))
### set up key-index pairs for yhat
combined_ids = {"yhat_wue": 15, "yhat_esi": 16, "yhat_agb": 17}
for ii in combined_ids:
    print(combined_min[combined_ids[ii]], combined_max[combined_ids[ii]])

### load relevent files for this analysis
for rkey in ad2:
    print("loading from", ad2[rkey]["loc"])
    raster_i = gdal.Open(ad2[rkey]["loc"])
    rasterband_i = raster_i.GetRasterBand(1)
    ### save NDV
    ad2[rkey]["ndv"] = rasterband_i.GetNoDataValue()
    ### save raster size
    ad2[rkey]["rs"] = (raster_i.RasterXSize, raster_i.RasterYSize)
    #tulh, tpxh, _, tulv, _, tpxv
    # tpxv -> abs (tpxv)
    geo_i = raster_i.GetGeoTransform()
    ### save crs
    ad2[rkey]["crs"] = (geo_i[0], geo_i[3], geo_i[1], abs(geo_i[5]))
    ### save data
    ad2[rkey]["data"] = np.array(raster_i.ReadAsArray().transpose())
    print(" -> read array", ad2[rkey]["data"].shape)
    del rasterband_i
    del raster_i

### scale data values --- formula is [S * (max-min)] + min
for ii in combined_ids:
    ad2[ii]["data"] = (ad2[ii]["data"] * (combined_max[combined_ids[ii]] - combined_min[combined_ids[ii]])) + combined_min[combined_ids[ii]]


### pretty sure that []_list is shape (n, 2) where (i, 0) is x coord etc.
def idx_to_geo_vec(idx_list, geopack, usetype=float):
    ulh, ulv, psh, psv = geopack
    
    geo_ul = np.array(idx_list).astype(float)
    geo_ul[:,0] = (geo_ul[:,0] * psh) + ulh
    geo_ul[:,1] = (-geo_ul[:,1] * psv) + ulv

    return geo_ul.astype(usetype)

def geo_to_idx_vec(geo_list, geopack, usetype=float):
    ulh, ulv, psh, psv = geopack

    idx_ul = np.array(geo_list).astype(float)

    idx_ul[:,0] = (idx_ul[:,0] - ulh) / psh
    idx_ul[:,1] = (ulv - idx_ul[:,1]) / psv

    return idx_ul.astype(usetype)

### setup
xkeys = ["lf_cc", "lf_evc", "standage"]
ykeys = ["yhat_wue", "yhat_esi", "yhat_agb"]
corekey = "lf_evt"
### get list of positions where core value is not ndv
corenum = np.argwhere(ad2[corekey]["data"] != ad2[corekey]["ndv"])
### sampling resolution calculation
standard_res = 30
sampling_res = 10
sr_sr = standard_res/sampling_res
comps = []

### comp is output
### format of comp is
### [temp[],
###  ...]

### what is this... make array of sampling_res/standard res ratio...
#usesss = np.array(np.array([[sampling_res/standard_res, sampling_res/standard_res]]))
### this is a vector to multiply comp_range by
### usesss = np.array([[sampling_res/standard_res, sampling_res/standard_res]])
### set up range from 0 to standard_res/sampling_res ratio
### comp_range = np.arange(int(sr_sr)).reshape(int(sr_sr), 1)
### offset is NOT USED!!!!
### comp_offset =  comp_range*usesss
### print(comp_offset)

print(len(corenum))
subsetting = 1

### subset? this is also not used rn
reduce_samplelocs = np.arange(len(corenum)//subsetting)
np.random.shuffle(reduce_samplelocs)
reduced_y = corenum[reduce_samplelocs]
print(len(corenum), len(reduce_samplelocs), len(reduced_y))

print(reduced_y[[10]], reduced_y[[10]] + np.array([0.5, 0.5]))

keycomp = [corekey]
if not savedflag:
    ### iterate over all subsetted samples
    for i in range(len(reduced_y)):
        ### print update
        if i % (len(reduced_y)//50) == 0:
            print(i/len(reduced_y))
        
        ### compute upper left of core sampling location from crs
        ts_ul = idx_to_geo_vec(reduced_y[[i]] + np.array([0.5, 0.5]), ad2[corekey]["crs"])
        ### extract the core data value at this location
        temps = [ad2[corekey]["data"][reduced_y[i][0], reduced_y[i][1]]]
        

        ndflag = True
        ### iterate over y layers
        for k in range(len(ykeys)):
            ### compute index upper left of y layer
            yk_ul = geo_to_idx_vec(ts_ul, ad2[ykeys[k]]["crs"], int)
            ### extract value at this position
            xtract_y = ad2[ykeys[k]]["data"][yk_ul[0, 0], yk_ul[0, 1]]
            ### verify this is not nodata
            if xtract_y != ad2[ykeys[k]]["ndv"]:
                temps.append(xtract_y)
                if i == 0:
                    keycomp.append(ykeys[k])
            else:
                ndflag = False

        ### iteate over x layers
        for j in range(len(xkeys)):
            ### compute index upper left of x layer
            xk_ul = geo_to_idx_vec(ts_ul, ad2[xkeys[j]]["crs"], int)
            ### extract value at this positon
            xtract_x = ad2[xkeys[j]]["data"][xk_ul[0, 0], xk_ul[0, 1]]
            ###... same
            if xtract_x != ad2[xkeys[j]]["ndv"]:
                temps.append(xtract_x)
                if i == 0:
                    keycomp.append(xkeys[j])
            else:
                ndflag = False
        ### use this flag to verify we have not found a nodata value at this position
        if ndflag:
            comps.append(temps)
            if len(comps[-1]) != 7:
                print(comps[-1])
                sys.exit(0)
    
    print("done")
    comps = np.array(comps)
    np.save('../visualize/geographic/comparicomputations', comps)
    print("saved")

comps = np.load('../visualize/geographic/comparicomputations.npy')
print("loaded")

print(keycomp)