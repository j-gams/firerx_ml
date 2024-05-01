### reworked helpers
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import os
import numpy as np

import math

def bil(filepath):
    return gdal.Open(filepath)

def tif(filepath):
    return gdal.Open(filepath)

def slice_batch(batch_params):
    band_list, layer_arr, resample_params = batch_params
    result = []
    for band in band_list:
        if band_list[0] == 0 and (band + 1) % (len(band_list) // 50) == 0:
            print("-", end="", flush=True)
        result.append(multiprocess_align(resample_params, layer_arr, band))
    if band_list[0] == 0:
        print("-->|  strip completed")
    return result

def chunk_list(length, nchunks, idx):
    return [ii for ii in range(idx * math.ceil(length / nchunks), min((idx + 1) * math.ceil(length / nchunks), length))]


def in_bounds_check(rastersize, idx_si, idx_sj):
    if idx_si < 0 or idx_sj < 0:
        return False
    if idx_si >= rastersize[0] or idx_sj >= rastersize[1]:
        return False
    return True

def multiprocess_align(params, layer_raster_np, band):
    ### provided params
    slice_width, sample_grid_size, layer_crs, target_crs, layer_nodata_val, oob_nodata_val, avg_method, sampling_method = params

    layer_ulh, layer_ulv, layer_ph, layer_pv = layer_crs
    target_ulh, target_ulv, target_ph, target_pv = target_crs

    ### computed values
    all_oob_count = 0
    temp_data = np.zeros(slice_width)
    ### eg 7/7/3 = 1/3 makes sense
    sampling_x_increment = (target_ph/sample_grid_size)/layer_ph
    sampling_y_increment = (target_pv/sample_grid_size)/layer_pv
    sampling_x_offset = sampling_x_increment/2
    sampling_y_offset = sampling_y_increment/2

    ### weighted average sampling arrays
    if sampling_method == "weighted_blocks":
        work_margin_i = ((target_ph - layer_ph) / layer_ph) % 1
        min_margin_i = 1 - work_margin_i
        work_margin_j = ((target_pv - layer_pv) / layer_pv) % 1
        min_margin_j = 1 - work_margin_j
        base_w_size_i = int(target_ph // layer_ph + 2)
        base_w_size_j = int(target_pv // layer_pv + 2)
        weights_i = np.ones((base_w_size_i, 1))
        weights_j = np.ones((1, base_w_size_j))
        oobmask = np.ones((base_w_size_i, base_w_size_j))

    ### idea... from target resolution, march through and sample
    ### target crs needs to have base y UL and target resolution pixel width

    ### slice is just one row of data. Iterate over and sample
    for j in range(slice_width):
        target_coord_i, target_coord_j = idx_to_coord(band, j, target_ulh, target_ulv, target_ph, target_pv)
        layer_idx_i, layer_idx_j = coord_to_idx(target_coord_i, target_coord_j, layer_ulh, layer_ulv, layer_ph, layer_pv)

        if sampling_method == "n_meter":
            ### perform slow sampling...
            sample = []
            ### do increment plus half of increment
            ### idea... compute fraction of index with increment/pixel width
            ### from upper left ... sample at floor(increment)
            for si in range(sample_grid_size):
                for sj in range(sample_grid_size):
                    idx_si = int(layer_idx_i + sampling_x_offset + (si * sampling_x_increment))
                    idx_sj = int(layer_idx_j + sampling_y_offset + (si * sampling_y_increment))
                    ### check out_of_bounds
                    if in_bounds_check(layer_raster_np.shape, idx_si, idx_sj):
                        retrieve = layer_raster_np[idx_si, idx_sj]
                        if retrieve != layer_nodata_val:
                            sample.append(retrieve)
            if len(sample) == 0:
                all_oob_count += 1
                temp_data[j] = oob_nodata_val
            else:
                if avg_method == "mean":
                    temp_data[j] = np.mean(np.array(sample))
                if avg_method == "mode":
                    vals, counts = np.unique(np.array(sample), return_counts=True)
                    temp_data[j] = vals[np.argwhere(counts == np.max(counts))][0]
        elif sampling_method == "weighted_blocks":
            margin_i = layer_idx_i % 1
            margin_j = layer_idx_j % 1
            if margin_i < min_margin_i:
                ### outside margins -- need to reduce size of sampler i
                temp_i = layer_idx_i % 1
                ###
                weights_i[0, 0] = 1 - temp_i
                ### end value is 0 since this is the small case
                weights_i[-1, 0] = 0
                ### this is the remainder of offset from UL plus margin
                weights_i[-2, 0] = (work_margin_i + layer_idx_i) % 1
            else:
                ### inside margins -- need to verify full size and interior 1s
                temp_i = layer_idx_i % 1
                weights_i[0, 0] = 1 - temp_i
                ###
                weights_i[-1, 0] = (work_margin_i + layer_idx_i) % 1
                weights_i[-2, 0] = 1
            if margin_j < min_margin_j:
                ### outside margins -- need to reduce size of sampler j
                temp_j = layer_idx_j % 1
                ###
                weights_j[0, 0] = 1 - temp_j
                ### end value is 0 since this is the small case
                weights_j[0, -1] = 0
                ### this is the remainder of offset from UL plus margin
                weights_j[0, -2] = (work_margin_j + layer_idx_j) % 1
            else:
                ### inside margins -- need to verify full size and interior 1s
                temp_j = layer_idx_j % 1
                weights_j[0, 0] = 1 - temp_j
                ###
                weights_j[0, -1] = (work_margin_j + layer_idx_j) % 1
                weights_j[0, -2] = 1
            oobmask[:, :] = 1
            ### for lower side... [0: max(0, 0 - UL)]
            lower_i = max(0, 0 - layer_idx_i)
            lower_j = max(0, 0-layer_idx_j)
            upper_i = min(-1, (layer_raster_np.shape[0]) - 2 - layer_idx_i)
            upper_j = min(-1, (layer_raster_np.shape[1]) - 2 - layer_idx_j)
            oobmask[0: int(lower_i), 0: int(lower_j)] = 0
            ### for higher side... [min(len-1, len + len - 2 - UL): len-1]
            oobmask[int(upper_i): layer_raster_np.shape[0], int(upper_j): layer_raster_np.shape[1]] = 0

            ### if there is overlap in either dim -- all oob
            if lower_i < layer_raster_np.shape[0] + upper_i or lower_j < layer_raster_np.shape[1] + upper_j:
                all_oob_count += 1
                ### set value to nodata
                temp_data[j] = oob_nodata_val

            else:
                ### determine where in the out of bounds mask area is not out of bounds
                countxy = np.where(oobmask != 0)
                ### retrieve non-oob data from layer raster
                temp = layer_raster_np[countxy[0] + math.floor(layer_idx_i), countxy[1] + math.floor(layer_idx_j)]
                ### if len is 0 it is all oob
                if len(temp) == 0:
                    all_oob_count += 1
                    temp_data[j] = oob_nodata_val
                ###
                elif avg_method == "mean":
                    ### matrix mult to determine weights, then weighted average
                    temp_data[j] = np.mean(temp * (weights_i @ weights_j)[countxy])
                elif avg_method == "mode":
                    unique_vals = np.unique(temp)
                    weighted_bin_count = np.bincount(np.searchsorted(unique_vals, temp), (weights_i @ weights_j)[countxy])
                    temp_data[j] = unique_vals[weighted_bin_count.argmax()]

    return temp_data

def idx_to_coord(ix, iy, ulh, ulv, psh, psv):
    cx = ulh + (ix * psh)
    cy = ulv - (iy * psv)
    return cx, cy

def coord_to_idx(cx, cy, ulh, ulv, psh, psv):
    ix = (cx - ulh) / psh
    iy = (ulv - cy) / psv
    return ix, iy

def convert_crs(from_cx, from_cy, from_ulh, from_ulv, from_psh, from_psv, to_ulh, to_ulv, to_psh, to_psv):
    ### convert crs... same projection.
    pass

def compute_new_crs():
    pass

def avg ():
    pass

def goofy_aws(params):
    pemdir, inst, instdir, asstr, index = params
    print("running set", index)
    os.system('nohup ssh -i "' + pemdir + '" ' + inst +
              ' "cd ~/' + instdir +
              '; nohup ~/anaconda3/bin/python align_raster.py core_aws=false core_aw2=true ' +
              'locl_ras=../' + instdir +
              '/ locl_tro=./geotifs_raw locl_rjo=geotifs_raw/ locl_clp=aligned_out skip_trm=true ' +
              'skip_rrj=true skip_erj=true skip_res=true skip_raw=true pram_sub=' + asstr +
              ' > ~/align_log.txt" > align_local_log.txt')