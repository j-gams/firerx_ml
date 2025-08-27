### geoplotting 2: plotting geo performance
### ground work for spatial plots for paper...
### need 'd2' environment for this not firerx
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from osgeo import gdal
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
import csv
import sys

### need to load in all the rasters and make nice images out of them jesus
combined_min = np.genfromtxt("../data/ml_sets/pyramid_v10/norm_layer_mins_combined.csv", delimiter=',')
combined_max = np.genfromtxt( "../data/ml_sets/pyramid_v10/norm_layer_maxs_combined.csv", delimiter=',')
combined_ids = {"yhat_wue": 15, "yhat_esi": 16, "yhat_agb": 17}
for ii in combined_ids:
    print(combined_min[combined_ids[ii]], combined_max[combined_ids[ii]])

### FIRST STEP IN RASTERIZING
geoperf_locs = ["../visualize/geographic/c2_late_a_pyramid_error_1_wue.tif",
                "../visualize/geographic/c2_late_a_pyramid_predictions_1_wue.tif",
                "../visualize/geographic/c2_late_a_pyramid_error_1_esi.tif",
                "../visualize/geographic/c2_late_a_pyramid_predictions_1_esi.tif",
                "../visualize/geographic/c2_late_a_pyramid_error_1_agb.tif",
                "../visualize/geographic/c2_late_a_pyramid_predictions_1_agb.tif"]
                
geoperf_rasters = []
geoperf_ndv = []
geoperf_quantiles = []

for gploc in geoperf_locs:
    print("loading from", gploc)
    gpraster = gdal.Open(gploc)
    gpband = gpraster.GetRasterBand(1)

    geoperf_ndv.append(gpband.GetNoDataValue())
    geoperf_rasters.append(np.array(gpraster.ReadAsArray().transpose()))
    print(" -> read array", geoperf_rasters[-1].shape)
    
    ### trim to remove extra ndv ...
    tempndv = np.argwhere(geoperf_rasters[-1] != geoperf_ndv[-1])
    print(" -> did argwheres")

    x_min, y_min = tempndv.min(axis=0)
    x_max, y_max = tempndv.max(axis=0)
    geoperf_rasters[-1] = geoperf_rasters[-1][x_min:x_max+1, y_min:y_max+1]
    print(" -> cropped to", geoperf_rasters[-1].shape)

combined_min = np.genfromtxt("../data/ml_sets/pyramid_v10/norm_layer_mins_combined.csv", delimiter=',')
combined_max = np.genfromtxt( "../data/ml_sets/pyramid_v10/norm_layer_maxs_combined.csv", delimiter=',')
print(len(combined_min), len(combined_max))
combined_ids2 = {1: 15, 
                 0: 15, 
                 3: 16, 
                 2: 16, 
                 4: 17, 
                 5: 17}
for ii in combined_ids:
    print(combined_min[combined_ids[ii]], combined_max[combined_ids[ii]])


for ii in range(len(geoperf_rasters)//2):
    ### actually, as far as error goes...
    ### if x in [0, 1] and error is (x1 - x2)^2
    ### now scale to [a, b]: error scales with (a-b)^2
    geoperf_rasters[(ii*2) + 1] = (geoperf_rasters[(ii*2) + 1] * (combined_max[combined_ids2[(ii*2) + 1]] - combined_min[combined_ids2[(ii*2) + 1]])) + combined_min[combined_ids2[(ii*2) + 1]]
    print(" -> scaled to", combined_min[combined_ids2[(ii*2)]], combined_max[combined_ids2[(ii*2)]])
    geoperf_rasters[(ii*2)] = geoperf_rasters[(ii*2)] * ((combined_max[combined_ids2[(ii*2)]] - combined_min[combined_ids2[(ii*2)]]) ** 2)
    
for ii in range(len(geoperf_rasters)):
    wdv = np.extract(geoperf_rasters[ii] != geoperf_ndv[ii], geoperf_rasters[ii])
    print("did wdv", len(wdv), geoperf_ndv[ii])
    wdv = wdv[~np.isnan(wdv)]
    #tq = geoperf_rasters[-1][tempndv]
    #print(" -> tq shape", tq.shape)
    geoperf_quantiles.append(np.quantile(wdv, [0, 0.25, 0.5, 0.75, 1]))
    print(geoperf_quantiles[ii])


### SECOND STEP HERE...

plt_titles = ["Late-Combination WUE Error",
              "Late-Combination WUE Predictions",
              "Late-Combination ESI Error",
              "Late-Combination ESI Predictions",
              "Late-Combination AGB Error",
              "Late-Combination AGB Predictions"]

plt_save = ["wue_error",
            "wue_predictions",
            "esi_error",
            "esi_predictions",
            "agb_error",
            "agb_predictions"]
#nice_y_names = ["Predicted Water Use Efficiency \n (g C kg$\mathregular{^{-1}}$ H$\mathregular{_2}$O)", 
#                "Predicted Evaporative Stress Index", "Predicted Above Ground Biomass \n (Mg ha$\mathregular{^{-1}}$)"]

plt_units = ["WUE MSE", "WUE Predictions (g C kg$\mathregular{^{-1}}$ H$\mathregular{_2}$O)", 
             "ESI MSE", "ESI Predictions", 
             "AGB MSE", "AGB Predictions (Mg ha$\mathregular{^{-1}}$)"]

textlabels = ["B", "A", "D", "C", "F", "E"]
textposs = [(100/7) * 500, (100/7) * 500, (100/7) * 500, (100/7) * 500, 500, 500]
textround = [4, 4, 4, 4, 6, 6]
labelpad = [-46, -45, -52, -45, -62, -60]

for i in range(len(geoperf_rasters)):
    #fig, ax = plt.subplots()
    plt.imshow(geoperf_rasters[i].transpose(), cmap='plasma_r', interpolation='nearest')
    plt.axis('off')
    plt.annotate(textlabels[i], xy=(0.85, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', horizontalalignment='right', fontfamily='serif',
        bbox=dict(facecolor='0.9', edgecolor='none', pad=3.0))
    
    ### try to do quantiles 
    keystr = "Maximum: " + str(round(geoperf_quantiles[i][4], textround[i])) + \
             "\nQ3: " + str(round(geoperf_quantiles[i][3], textround[i])) + \
             "\nMedian: " + str(round(geoperf_quantiles[i][2], textround[i])) + \
             "\nQ1: " + str(round(geoperf_quantiles[i][1], textround[i])) + \
             "\nMinimum: " + str(round(geoperf_quantiles[i][0], textround[i]))
    props = dict(boxstyle='round', facecolor='aliceblue', alpha=0.5)

    # place a text box in upper left in axes coords
    plt.text(textposs[i], 0.005, keystr, fontsize=6,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    #plt.title(plt_titles[i])
    cbar = plt.colorbar(fraction=0.025, pad=-0.15)
    cbar.set_label(plt_units[i], rotation=90, labelpad=labelpad[i])
    plt.savefig("../visualize/realgeoplots/" + plt_save[i] + ".png", bbox_inches = 'tight')
    plt.close()