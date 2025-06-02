### spatial graphs

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import math
sys.path.append("..")
sys.path.append("../models")
from models.data_handler import data_wrangler
import models.modelbase.mltools as mlt
from modelbase.cascade_late_a import model_cascade2_late_a
from modelbase.cascade_mid import model_cascade2_mid
from modelbase.cascade_early import model_cascade2_early
from modelbase.cascade_early_b import model_cascade2_early_b
from modelbase.vit_pyrencoder import multi_vit
from modelbase.cube_cnn import model_flat2

from osgeo import gdal

loc_prefix = "../models/trained/"
data_locs = ["../data/ml_sets/pyramid_v10/","../data/ml_sets/cube_v10/", "../data/ml_sets/adjust_v10/"]
raster_prefix = "../data/aligned_raster/colorado_ml_v10/"
raster_locs = ["ECOSTRESS_WUE_22_4326_0_0.tif", "ECOSTRESS_ESI_22_4326_0_0.tif", "GEDI_AGB_4326_0_0.tif"]
### model_tag: [model_dir, config_name, dataset, nice_name]
ops = {"c2_early_pyramid":  ["c2_early_b_pyramid",  "c2_early_b_pyramid",   "C2-Early (Pyramid)", model_cascade2_early(),     0],
       "c2_mid_pyramid":    ["c2_mid_pyramid",      "c2_mid_pyramid",       "C2-Mid (Pyramid)",   model_cascade2_mid(),       0],
       "c2_late_pyramid":   ["c2_late_a_pyramid",   "c2_late_a_pyramid",    "C2-Late (Pyramid)",  model_cascade2_late_a(),    0],
       "multivit_pyramid":  ["multi_vit_pyramid",   "multi_vit_pyramid",    "multi-ViT (Pyramid)",multi_vit(),                0],
       "c2_early_cube":     ["c2_early_b_cube",     "c2_early_b_cube",      "C2-Early (Cube)",    model_cascade2_early(),     1],
       "c2_mid_cube":       ["c2_mid_cube",         "c2_mid_cube",          "C2-Mid (Cube)",      model_cascade2_mid(),       1],
       "c2_late_cube":      ["c2_late_a_cube",      "c2_late_a_cube",       "C2-Late (Cube)",     model_cascade2_late_a(),    1],
       "multivit_cube":     ["multi_vit_cube",      "multi_vit_cube",       "multi-ViT (Cube)",   multi_vit(),                1],
       "c2_early_adj":      ["c2_early_b_adjust",   "c2_early_b_adjust"     "C2-Early (Adj)",     model_cascade2_early(),     2],
       "c2_mid_adj":        ["c2_mid_adjust",       "c2_mid_adjust",        "C2-Mid (Adj)",       model_cascade2_mid(),       2],
       "c2_late_adj":       ["c2_late_a_adjust",    "c2_late_a_adjust",     "C2-Late (Adj)",      model_cascade2_late_a(),    2],
       "multivit_adj":      ["multi_vit_adjust",    "multi_vit_adjust",     "multi-ViT (Adj)",    multi_vit(),                2], 
       "singletask_wue":    ["f2_singletask_0",     "c2_singletask_0",      "C2-Single (WUE)",    model_cascade2_late_a(),    1],
       "singletask_esi":    ["f2_singletask_1",     "c2_singletask_0",      "C2-Single (ESI)",    model_cascade2_late_a(),    1],
       "singletask_agb":    ["f2_singletask_2",     "c2_singletask_0",      "C2-Single (AGB)",    model_cascade2_late_a(),    1]}

runon = ["c2_early_pyramid",
         "c2_mid_pyramid",
         "c2_late_a_pyramid",
         "multi_vit_pyramid",
         
         "c2_early_cube",
         "c2_mid_cube",
         "c2_late_a_cube",
         "multi_vit_cube",
         "f2_baseline_cube",
         
         "c2_early_adjust",
         "c2_mid_adjust",
         "c2_late_a_adjust",
         "multi_vit_adjust",
         "f2_baseline_adjust"]

runon = ["singletask_wue"]

data_locs = ["../data/ml_sets/pyramid_v10/",
             "../data/ml_sets/cube_v10/",
             "../data/ml_sets/adjust_v10/"]

y_names = [["Water Use Efficiency", "wue"],
           ["Evaporative Stress Index", "esi"],
           ["Above Ground Biomass", "agb"]]

### load metadata
### load data

def dataload(datalocs, dli):

    print("loading data with dataload")
    metadata = mlt.load_metadata(datalocs[dli])
    metadata = metadata
    wrangler = data_wrangler(datalocs[dli], metadata["n_layers"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                   metadata["layer_dims"], 1000, metadata["buffer_nodata"], 
                                   metadata["x_layers"], metadata["y_layers"],  sample_weights=False,
                                   low_mem=False)
    wrangler.setup("val")
    temp_li = np.genfromtxt(datalocs[dli] + "/legal_ids.csv", delimiter=",").astype(int)
    #legalids = temp_li
    print(temp_li.shape)
    return metadata, wrangler, temp_li

working_crs = []


### need to get shapes, crs of original data...
y_geolayers = []
for i in range(3):
    raster_i = gdal.Open(raster_prefix+raster_locs[i])
    rasterband_i = raster_i.GetRasterBand(1)
    nodata_i = rasterband_i.GetNoDataValue()
    rastersize_i = (raster_i.RasterXSize, raster_i.RasterYSize)
    #tulh, tpxh, _, tulv, _, tpxv
    # tpxv -> abs (tpxv)
    geo_i = raster_i.GetGeoTransform()
    proj_i = raster_i.GetProjection()
    working_crs.append((geo_i[0], geo_i[1], geo_i[3], abs(geo_i[5])))
    del rasterband_i
    del raster_i

    y_geolayers.append([nodata_i, rastersize_i, geo_i, proj_i])

#####
def read_config(in_loc):
    config_in = open(in_loc)
    config_dict = json.load(config_in)
    return config_dict

fold_subset = [0]

guiding = 2
block_dims = [15, 15, 1]
block_res = [70, 70, 1000]

### convert from coordinates to indices -- from create_pyramid_functions
def geo_idx(cx, cy, geopack): #ulh, ulv, psh, psv):
    ulh, ulv, psh, psv = geopack
    ix = (cx - ulh) / psh
    iy = (ulv - cy) / psv
    return ix, iy

### convert from indices to coordinates -- same...
def idx_geo(ix, iy, geopack): #ulh, ulv, psh, psv):
    ulh, ulv, psh, psv = geopack
    cx = ulh + (ix * psh)
    cy = ulv - (iy * psv)
    return cx, cy

# metadatas
# wranglers
# legalids

### load models
for mk in runon:
    print("evaluating model", mk)
    ### load config...
    print(loc_prefix + ops[mk][0] + "/" + ops[mk][1] + ".json")
    model_config_i = read_config(loc_prefix + ops[mk][0] + "/" + ops[mk][1] + ".json")
    ### run params
    run_params = model_config_i["run_params"]
    ### model params
    model_parameters = model_config_i["model_params"]

    ### make predictions, compute MSE over all points
    for fold in fold_subset:
        ### nead to load wrangler, metadata, etc. to begin with
        meta_mk, wrangler_mk, legal_mk = dataload(data_locs, ops[mk][-1])
        ### ....
        print("fold", fold, end="...", flush=True)
        ### set wrangler to appropriate fold...
        wrangler_mk.set_fold(fold)
        ### setup model
        working_model = ops[mk][3]
        working_name = run_params["model_name"] + "_" + str(fold)
        working_model.setup(model_parameters["hyperparams"], meta_mk, "../models/" + run_params["model_dir"], working_name,
                                    run_params["train_params"]["verbosity"], run_params["train_params"]["callbacks"])
        ### load model from file...
        working_model.load()
        print("loaded model", end="...", flush=True)
        ### make predictions
        y_a, y_hat = working_model.predict(wrangler_mk)
        print("made predictions", end="...", flush=True)
        ### compute square error over all points
        ### this should come out as [3][n_pts...]
        y_mse = mlt.metric_mse(y_hat, y_a, "flattened", ["each"])
        print("step a")
        ### make empty raster
        spatial_error = []

        ### convert shuffled ids to coords in raster space
        val_to_int = wrangler_mk.val_ids[fold].copy()
        val_to_int = val_to_int.astype(int)
        coords_from_ids = legal_mk[val_to_int]
        print("step b")

        print(y_geolayers[0][1][0] * block_dims[0])

        ###
        for i in range(3):
            spatial_error.append(np.zeros((y_geolayers[i][1][0], y_geolayers[i][1][1])))
            spatial_error[-1] += y_geolayers[i][0]

        print("step c")

        print("debugging")
        print(val_to_int.shape)
        print(val_to_int[:5])
        
        print(coords_from_ids[:5])
        print(y_mse[0][2])
        y_mse_r = y_mse[0][2].reshape((y_mse[0][2].shape[0], int(math.sqrt(y_mse[0][2].shape[1])), 
                                       int(math.sqrt(y_mse[0][2].shape[1]))))
        print(y_mse_r.shape)
        print(y_mse[0][2].shape)
        print(len(y_mse))
        print(y_mse[0][2][7].shape)
        print("deleting meta, wrangler, legal")

        del meta_mk
        
        del legal_mk

        for j in range(len(wrangler_mk.val_ids[fold])):
            for i in range(3):
                ### coords in legal_samples wrt sampling grid 
                ### -- which is in the guiding crs. This is the AGB 1000m crs.
                ### -- need to...
                suli = coords_from_ids[j][0]
                sulj = coords_from_ids[j][1]
                if i < 2:
                    ### convert to general crs...
                    cx, cy = idx_geo(suli + 0.5, sulj + 0.5, working_crs[guiding])
                    ### convert back to this layer crs... 
                    li, lj = geo_idx(cx, cy, working_crs[i])
                    ### compute the ultimate UL index for this sample
                    ### round down to get central pixel UL
                    ### ... this is assuming odd dim sizes
                    ### subtract dim//2 to get ultimate UL
                    suli = int(li) - (block_dims[i] + 1) % 2 #(block_dims[i] // 2)
                    sulj = int(lj) - (block_dims[i] + 1) % 2 #(block_dims[i] // 2)
                    if suli+block_dims[i] >= spatial_error[i].shape[0] or sulj+block_dims[i] >= spatial_error[i].shape[1]:
                        print(spatial_error[i].shape)
                        print("UH OH...", suli+block_dims[i], sulj+block_dims[i], block_dims[i])
                spatial_error[i][suli:suli+block_dims[i], sulj:sulj+block_dims[i]] = y_mse[0][i][j].reshape((block_dims[i], block_dims[i]))

        del wrangler_mk
        #print("breaking")
        #break
            
        driver = gdal.GetDriverByName("GTiff")
        outname_base = "../visualize/geographic/"
        outname_mods = ["_wue", "_esi", "_agb"]
        for i in range(3):
            out_i = driver.Create(outname_base + ops[mk][0] + "_" + str(fold) + outname_mods[i] + ".tif",
                                  spatial_error[i].shape[0], spatial_error.shape[1], 1, gdal.GDT_Float32)
            out_i.SetGeoTransform(y_geolayers[i][2])
            out_i.SetProjection(y_geolayers[i][3])
            out_i.GetRasterBand(1).WriteArray(spatial_error[i].transpose())
            out_i.GetRasterBand(1).SetNoDataValue(y_geolayers[i][0])
            out_i.FlushCache()
            print("converted", outname_mods[i], end="...", flush=True)
        print("complete")


