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

combined_min = np.genfromtxt("../data/ml_sets/pyramid_v10/norm_layer_mins_combined.csv", delimiter=',')
combined_max = np.genfromtxt( "../data/ml_sets/pyramid_v10/norm_layer_maxs_combined.csv", delimiter=',')
print(len(combined_min), len(combined_max))
combined_ids = {"yhat_wue": 15, "yhat_esi": 16, "yhat_agb": 17}
for ii in combined_ids:
    print(combined_min[combined_ids[ii]], combined_max[combined_ids[ii]])

### load relevent files for this analysis
for rkey in ad2:
    print("loading from", ad2[rkey]["loc"])
    raster_i = gdal.Open(ad2[rkey]["loc"])
    rasterband_i = raster_i.GetRasterBand(1)
    ad2[rkey]["ndv"] = rasterband_i.GetNoDataValue()
    ad2[rkey]["rs"] = (raster_i.RasterXSize, raster_i.RasterYSize)
    #tulh, tpxh, _, tulv, _, tpxv
    # tpxv -> abs (tpxv)
    geo_i = raster_i.GetGeoTransform()
    ad2[rkey]["crs"] = (geo_i[0], geo_i[3], geo_i[1], abs(geo_i[5]))
    ad2[rkey]["data"] = np.array(raster_i.ReadAsArray().transpose())
    print(" -> read array", ad2[rkey]["data"].shape)
    del rasterband_i
    del raster_i

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
corenum = np.argwhere(ad2[corekey]["data"] != ad2[corekey]["ndv"])
standard_res = 30
sampling_res = 10
sr_sr = standard_res/sampling_res
comps = []

usesss = np.array(np.array([[sampling_res/standard_res, sampling_res/standard_res]]))
comp_range = np.arange(int(sr_sr)).reshape(int(sr_sr), 1)
comp_offset =  comp_range*usesss

print(len(corenum))
subsetting = 1

reduce_samplelocs = np.arange(len(corenum)//subsetting)
np.random.shuffle(reduce_samplelocs)
reduced_y = corenum[reduce_samplelocs]
print(len(corenum), len(reduce_samplelocs), len(reduced_y))

keycomp = [corekey]
if not savedflag:
    for i in range(len(reduced_y)):
        if i % (len(reduced_y)//50) == 0:
            print(i/len(reduced_y))
        
        ts_ul = idx_to_geo_vec(reduced_y[[i]], ad2[corekey]["crs"])
        temps = [ad2[corekey]["data"][reduced_y[i][0], reduced_y[i][1]]]
        

        ndflag = True
        for k in range(len(ykeys)):
            yk_ul = geo_to_idx_vec(ts_ul, ad2[ykeys[k]]["crs"], int)
            xtract_y = ad2[ykeys[k]]["data"][yk_ul[0, 0], yk_ul[0, 1]]
            if xtract_y != ad2[ykeys[k]]["ndv"]:
                temps.append(xtract_y)
                if i == 0:
                    keycomp.append(ykeys[k])
            else:
                ndflag = False

        for j in range(len(xkeys)):
            xk_ul = geo_to_idx_vec(ts_ul, ad2[xkeys[j]]["crs"], int)
            xtract_x = ad2[xkeys[j]]["data"][xk_ul[0, 0], xk_ul[0, 1]]
            if xtract_x != ad2[xkeys[j]]["ndv"]:
                temps.append(xtract_x)
                if i == 0:
                    keycomp.append(xkeys[j])
            else:
                ndflag = False
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