from osgeo import gdal
import numpy as np
import h5py
import os
import math
import raster_helpers
from scipy.signal import convolve2d
from multiprocessing import Pool

### BASIC DATA PROCESSING PREPARATION
###
###

### compute expected cube size
def compute_expected_cube_sizes(y_layer_max, cube_res):
    expected_cube_size = []
    for i in range(len(cube_res)):
        if cube_res[i] == y_layer_max:
            expected_cube_size.append(1)
        else:
            expected_cube_size.append(y_layer_max // cube_res[i] + 1)
    print("- computed expected cube sizes: ", expected_cube_size)
    return expected_cube_size

### make buffer around smaller data layers
def make_buffer(buffer_fill, layer_data, cube_res, dimension_override):
    buffer_dist = []
    for i in range(len(cube_res)):
        buffer_dist.append(dimension_override//cube_res[i] + 1)
        layer_data[i] = np.pad(layer_data[i], (buffer_dist[i], buffer_dist[i]), mode="constant", constant_values=buffer_fill[i])
    print("- computed buffer distances: ", buffer_dist)
    return buffer_dist, layer_data

def reduce_from_total_dims(expected_cube_size, expected_sample_to, x_layers, reduce_factor):
    ### certainly a smarter way to do this but the point is to mirror the total number of params
    ### in the data pyramid but in a cube
    total_params_cube = 0
    total_params_sample = 0
    for i in range(len(expected_sample_to)):
        if i in x_layers:
            total_params_cube += expected_cube_size[i] ** 2
            total_params_sample += expected_sample_to[i] ** 2
    ratio = math.sqrt(total_params_cube / total_params_sample)
    for i in range(len(expected_sample_to)):
        if i in x_layers:
            expected_sample_to[i] = int(expected_sample_to[i] * ratio * reduce_factor)
    print("- reduced sampling dimension to match natural total parameters:", expected_sample_to[0])
    print("- using reduction factor:", reduce_factor)
    print("- new sample dimensions: ", expected_sample_to)
    return expected_sample_to


### compute offsets for sample generation
def compute_offsets(expected_cube_size, layer_data):
    center_offset = []
    half_offset = []
    for i in range(len(layer_data)):
        center_offset.append(expected_cube_size[i] // 2)
        if expected_cube_size[i] % 2 == 0:
            half_offset.append(0.5)
        else:
            half_offset.append(0)
    print("- computed sampling offsets")
    return center_offset, half_offset

### DEALING WITH REFERENCE SYSTEMS
###
###

### convert from coordinates to indices
def geo_idx(cx, cy, geopack): #ulh, ulv, psh, psv):
    ulh, ulv, psh, psv = geopack
    ix = (cx - ulh) / psh
    iy = (ulv - cy) / psv
    return ix, iy

### convert from indices to coordinates
def idx_geo(ix, iy, geopack): #ulh, ulv, psh, psv):
    ulh, ulv, psh, psv = geopack
    cx = ulh + (ix * psh)
    cy = ulv - (iy * psv)
    return cx, cy

### ROUNDUP 1: FIND LEGAL SAMPLES
###
###

### determine whether a sample is legal
### legality requires no missing data in the sample area
def roundup_layer_1a(k, base_idx, buffer_ignore, crs_list, y_base, nd_vals, expected_cube_size,
                    layer_data, buffer_dist, half_offset, center_offset, base_scaled_crs):
    ### retrieve sample-resolution coordinates
    ### these ids are wrt guiding UL (no buffer offset)
    bi, bj = base_idx
    ### convert to base layer ids
    bi *= base_scaled_crs
    bj *= base_scaled_crs
    ### get coords from base crs
    geo_ctri, geo_ctrj = idx_geo(bi + 0.5, bj + 0.5, crs_list[y_base])
    ### convert to ids in layer crs
    tidi, tidj = geo_idx(geo_ctri, geo_ctrj, crs_list[k])
    ### convert to layer grid space (UL) + offset
    sulx = int(tidi + half_offset) - center_offset
    suly = int(tidj + half_offset) - center_offset
    ###
    gather = layer_data[sulx+buffer_dist: sulx+buffer_dist+expected_cube_size,
                        suly+buffer_dist: suly+buffer_dist+expected_cube_size].reshape(-1)
    if len(gather) < expected_cube_size*expected_cube_size:
        print("caught")
        return False
    if np.isnan(gather).any():
        return False
    if buffer_ignore in gather:
        return False
    if nd_vals in gather:
        return False
    return True

def compile_legal_samples(expected_cube_size, layer_data, y_base, base_res, dimension_override, buffer_fill,
                          layer_crs, layer_nodata, buffer_dist, half_offset, center_offset, layer_size):
    ### gather legal samples
    ### ...batch based on regions?
    legal_sample_idx_list = []
    ### this is the shape of the y_base layer, (the layer crs we align to)
    guide_layer_shape = layer_data[y_base].shape
    ### now determine the shape of the layer if we use dim_override resolution
    ### as in --- the data at the desired sampling resolution
    ### this res factor is sample_dim / base_dim, so divide for grid size and multiply for crs resolution
    sample_res_factor = dimension_override / base_res[y_base]

    ### subtract the buffer dist...
    override_shape = (int(((guide_layer_shape[0] - (2*buffer_dist[y_base])) / sample_res_factor)),
                      int(((guide_layer_shape[1] - (2*buffer_dist[y_base])) / sample_res_factor)))

    #base_scaled_crs = (layer_crs[y_base][0], layer_crs[y_base][1], layer_crs[y_base][2] * sample_res_factor, layer_crs[y_base][3] * sample_res_factor)
    ### TODO -- rename if this works
    base_scaled_crs = sample_res_factor
    ### perform quick sanity check...
    falsecount = 0
    ### from here override_shape is confined to tbe region of base layer within buffer zone but other layers have buffer
    print("- compile samples progress ", end="", flush=True)
    ### iterate over the sampling-resolution grid and check whether there is any missing data or other issues in sample
    for i in range(override_shape[0]):
        ### just some user-friendly stuff for the impatient like me
        if i % (override_shape[0] // 20) == 0:
            print("-", end="", flush=True)
        if i == override_shape[0] - 1:
            print("->| done")
        for j in range(override_shape[1]):
            ### determine if nodata value is involved, and ignore buffer fill values
            all_ok = True
            ### within this sample region, check each cube for missing data etc
            for k in range(len(base_res)):
                ### check individual layer for nodata/NaN values within sample region
                """layer_k_np = roundup_layer_1a(k, (i + buffer_dist[y_base], j + buffer_dist[y_base]), buffer_fill[k],
                                              layer_crs, y_base, layer_nodata[k], expected_cube_size[k], layer_data[k],
                                              buffer_dist[k], half_offset[k], center_offset[k], base_scaled_crs)"""
                ### option without buffer dist...
                layer_k_np = roundup_layer_1a(k, (i, j), buffer_fill[k],
                                              layer_crs, y_base, layer_nodata[k], expected_cube_size[k], layer_data[k],
                                              buffer_dist[k], half_offset[k], center_offset[k], base_scaled_crs)
                ### if one layer is missing data, we have a problem
                if not layer_k_np:
                    all_ok = False
                    break
            ### if all layers are ok within sample area, this is a legal sample
            if all_ok:
                legal_sample_idx_list.append((i, j))
            else:
                falsecount += 1
    print("- rounded up layers: ", len(legal_sample_idx_list))
    print("  - sampling at override dimension: ", dimension_override, " with factor: ", sample_res_factor)
    print("  - maximum legal samples = ", override_shape[0] * override_shape[1])
    print("  -", falsecount, "rejected (" + str(falsecount/(override_shape[0]*override_shape[1])) + ")")
    return legal_sample_idx_list, override_shape, sample_res_factor

### save list of legal samples for future use
def save_legal_sample_ids(legal_sample_idx_list, fold_name):
    legal_idx_save = np.array(legal_sample_idx_list)
    np.savetxt(fold_name + "/legal_ids.csv", legal_idx_save, delimiter=",")
    print("- saved sample coords")

### load list of legal samples
def load_legal_sample_ids(fold_name, to_int=True):
    loaded_ids = np.genfromtxt(fold_name + "/legal_ids.csv", delimiter=",")
    if to_int:
        loaded_ids = loaded_ids.astype(int)
    return loaded_ids
### TRAIN/VAL/TEST SPLIT METHODS
###
###

def create_block_buffer_mask(block_mask, buffer):
    ### 0 for illegal, 1 for legal, 2 for taken
    ### set 1-values to 0 and non-1-values to 1, then run a kernel over
    ### copy the block mask
    block_buffer_mask = np.array(block_mask)
    ### set taken values to 1 and all other values to 0
    block_buffer_mask[block_mask != 2] = 0
    block_buffer_mask[block_mask == 2] = 1
    ### kernel to set square to off limits if anything within buffer, buffer of a square is taken
    kernelrc = np.ones(((buffer * 2) + 1, (buffer * 2) + 1))
    ### apply kernel to the mask
    bbm_1 = np.int64(convolve2d(block_buffer_mask, kernelrc, mode='same') > 0)
    ### now set all illegal points to off-limits (we don't need a buffer around them)
    bbm_1[block_mask == 0] = 1
    ### obtain a list of all locations where we can sample from (not taken, outside buffer, not illegal)
    sample_from = np.where(bbm_1 == 0)
    ### if there are no samples we have a problem....
    if len(sample_from[0]) == 0:
        ### no samples to go from! uh oh!
        return False, False
    ### return the list of legal samples and the block mask with buffer
    return sample_from, bbm_1



### TODO - CHECK RETURNS
### mask legal samples for test set
def block_split_test_iter(split_blocks_nregions, guide_shape, blocksizes, block_mask_in, split_blocks_buffer,
                          each_err_test, approx_test_each):
    ### copy original mask to avoid weird referencing errors
    block_mask = block_mask_in.copy()
    ### place blocks over region
    ### update - make better block mask
    nplaced = 0
    carrier = 0
    print("- test creation progress ", end="", flush=True)
    ### keep
    while nplaced < split_blocks_nregions:
        print("iter")
        ### narrow list of good options to randomly sample from
        narrow_list, bbm = create_block_buffer_mask(block_mask, split_blocks_buffer)
        if narrow_list == False:
            print("- oops! no legal locations to begin drawing boxes from. Restarting...")
            return block_split_test_iter(split_blocks_nregions, guide_shape, blocksizes, block_mask,
                                         split_blocks_buffer, each_err_test, approx_test_each)
        idrand = int(np.random.uniform(0, len(narrow_list[0])))
        ruli = narrow_list[0][idrand]
        rulj = narrow_list[1][idrand]
        if block_mask[ruli, rulj] == 1:
            nplaced += 1
            if nplaced % (split_blocks_nregions // 10) == 0:
                print("-", end="", flush=True)
            carrier += each_err_test
            reqd = int(approx_test_each) + int(carrier)
            carrier = carrier % 1
            ### good to go! start with centerpoint
            total1 = 0
            tradius = -1
            iteri = (int(np.random.uniform(0, 2)) * 2) - 1
            iterj = (int(np.random.uniform(0, 2)) * 2) - 1
            while total1 < reqd:
                tradius += 2
                ### need to account for oob
                for i in range(tradius):
                    for j in range(tradius):
                        ti = ruli - (iteri * (tradius // 2)) + (iteri * i)
                        tj = rulj - (iterj * (tradius // 2)) + (iterj * j)
                        if ti >= 0 and ti < block_mask.shape[0] and tj >= 0 and tj < block_mask.shape[1]:
                            if block_mask[ti, tj] == 1:
                                total1 += 1
                                block_mask[ti, tj] = 2
                                if total1 >= reqd:
                                    break
                    if total1 >= reqd:
                        break
        else:
            print("x", block_mask[ruli, rulj], bbm[ruli, rulj], end="", flush=True)
            #pass
    print(">| done")
    return block_mask

### mask legal samples for train/val sets
def block_split_tranval_iter(block_mask, split_blocks_nregions, split_outer_buffer, guide_shape,
                             split_blocks_buffer, blocksizes, each_err_val, approx_val_each):
    split_mask_i = block_mask.copy()
    ### place blocks over region
    nplaced_i = 0
    carrier = 0
    ### while the number of blocks placed is less than the number of desired regions
    while nplaced_i < split_blocks_nregions:
        ### obtain a list of samples that are not taken, not within a buffer of taken samples, and not illegal
        ### also bbm is the geographic mask
        narrow_list, bbm = create_block_buffer_mask(split_mask_i, split_blocks_buffer)
        ### if we don't find any samples...
        if narrow_list == False:
            print("- oops! no legal locations to begin drawing boxes from. Restarting...")
            return block_split_test_iter(split_blocks_nregions, guide_shape, blocksizes, block_mask,
                                         split_blocks_buffer, each_err_val, approx_val_each)
        ### uniformly sample a centerpoint coordinate from the narrow list
        idrand = int(np.random.uniform(0, len(narrow_list[0])))
        ruli = narrow_list[0][idrand]
        rulj = narrow_list[1][idrand]
        ### if we find a 1 at this location (legal sample), place a block around this point
        if split_mask_i[ruli, rulj] == 1:
            nplaced_i += 1
            if nplaced_i % (split_blocks_nregions // 10) == 0:
                print("-", end="", flush=True)
            carrier += each_err_val
            reqd = int(approx_val_each) + int(carrier)
            carrier = carrier % 1
            ### good to go! start with centerpoint
            total1 = 0
            tradius = -1
            iteri = (int(np.random.uniform(0, 2)) * 2) - 1
            iterj = (int(np.random.uniform(0, 2)) * 2) - 1
            while total1 < reqd:
                tradius += 2
                ### need to account for oob
                for i in range(tradius):
                    for j in range(tradius):
                        ti = ruli - (iteri * (tradius // 2)) + (iteri * i)
                        tj = rulj - (iterj * (tradius // 2)) + (iterj * j)
                        if ti >= 0 and ti < split_mask_i.shape[0] and tj >= 0 and tj < split_mask_i.shape[1]:
                            if split_mask_i[ti, tj] == 1:
                                total1 += 1
                                split_mask_i[ti, tj] = 2
                                if total1 >= reqd:
                                    break
                    if total1 >= reqd:
                        break
        else:
            print("x", split_mask_i[ruli, rulj], bbm[ruli, rulj], end="")
            #pass
    print(">| done")
    return split_mask_i

### Split in geographic blocks
def block_split(legal_sample_idx_list, partition, split_blocks_nregions, guide_shape, split_blocks_buffer, fold_name,
                layer_proj, y_base, n_splits, split_outer_buffer):
    ### first do some background work
    ### initialize lists for train/val partition, then compute the number of test samples
    val_fold_indices = []
    train_fold_indices = []
    n_test_samples = int(len(legal_sample_idx_list) * partition[0])
    ### this is block-based splitting, so compute how many samples per block based on number of blocks
    approx_each = n_test_samples / split_blocks_nregions
    ### aprroximate side length of each block
    temp_each_sqrt = math.ceil(math.sqrt(approx_each))

    ### WHAT DO THESE DOOOOOO
    minerr = 100000
    minoff = 0

    ### ????
    ### computing errors and offsets, but to what end?
    for i in range(max(temp_each_sqrt // 5, 1)):
        err1 = abs(
            (math.ceil(approx_each / (temp_each_sqrt - (i - (temp_each_sqrt // 10)))) * temp_each_sqrt) - approx_each)
        if err1 < minerr:
            minerr = err1
            minoff = i - (temp_each_sqrt // 10)

    ### ....?
    blocksizes = (temp_each_sqrt - minoff, math.ceil(approx_each / (temp_each_sqrt - minoff)))
    underest = (blocksizes[0] * blocksizes[1]) - (n_test_samples / split_blocks_nregions)
    intoffset = underest - int(underest)
    n_val_samples = int(len(legal_sample_idx_list) * partition[1])
    approx_val_each = n_val_samples / split_blocks_nregions
    each_err_val = approx_val_each - int(approx_val_each)

    n_test_samples = int(len(legal_sample_idx_list) * partition[0])
    approx_test_each = n_test_samples / split_blocks_nregions
    each_err_test = approx_test_each - int(approx_test_each)

    print("- computed test set block sizes")
    print("  - total_blocks:", n_test_samples, "approx. each:", approx_each, "sqrt:", temp_each_sqrt)
    print("  - blocksizes:", blocksizes, "block each:", blocksizes[0]*blocksizes[1], "est:", underest)
    print("  - guide_shape", guide_shape)

    ### initialize mask grid
    block_mask = np.zeros(guide_shape)
    ### initialize metaindex grid
    metaindex_arr = np.zeros(guide_shape)
    ### iterate over legal samples and set block_mask to 1 at legal locations
    ### ... and set metaindex_arr to the location index at legal locations
    for i in range(len(legal_sample_idx_list)):
        ti, tj = legal_sample_idx_list[i]
        block_mask[ti, tj] = 1
        metaindex_arr[ti, tj] = i

    print("- set up mask layers (step 1)")
    block_mask = block_split_test_iter(split_blocks_nregions, guide_shape, blocksizes, block_mask, split_blocks_buffer,
                          each_err_test, approx_test_each)
    print("- set up mask layers (step 2)")
    overlap3 = np.where(block_mask == 2)
    test_indices = metaindex_arr[overlap3]
    remaining_indices = metaindex_arr[np.where(block_mask == 1)]
    print("- found test indices:", len(test_indices), "out of", n_test_samples, "desired")

    ### make tifs for visual confirmation
    os.system("rm " + fold_name + "/fold_box_geotifs/*")
    os.system("mkdir " + fold_name + "/fold_box_geotifs")
    raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "test_extent", block_mask, layer_proj[y_base][0],
                               layer_proj[y_base][1], -1)

    print("- saved test geotif visualizations")

    block_mask[overlap3] = 0

    ### now do xval-splits
    for ii in range(n_splits):
        print("- fold", ii, "creation progress ", end="")
        split_mask_i = block_split_tranval_iter(block_mask, split_blocks_nregions, split_outer_buffer,
                                                guide_shape, split_blocks_buffer, blocksizes, each_err_val,
                                                approx_val_each)
        overlap4 = np.where(split_mask_i == 2)
        val_fold_indices.append(metaindex_arr[overlap4])
        train_fold_indices.append(metaindex_arr[np.where(split_mask_i == 1)])
        np.savetxt(fold_name + "/train_" + str(ii) + ".csv", train_fold_indices[-1], delimiter=",")
        np.savetxt(fold_name + "/val_" + str(ii) + ".csv", val_fold_indices[-1], delimiter=",")

        ### make geotif for visual inspection
        raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "val_extent_" + str(ii), split_mask_i,
                                   layer_proj[y_base][0], layer_proj[y_base][1], -1)
    print("- found training and validation indices")
    print("- saved training and validation indices")
    print("- saved training and validation geotif visualizations")
    np.savetxt(fold_name + "/test.csv", test_indices, delimiter=",")
    np.savetxt(fold_name + "/remaining.csv", remaining_indices, delimiter=",")
    print("- saved test indices")

    return test_indices, remaining_indices, train_fold_indices, val_fold_indices

def quick_block_iter(block_mask_in, n_blocks, n_samples, approx_each, cost_cap, initial_block_buffer):
    ### kernel to set square to off limits if anything within buffer, buffer of a square is taken
    kernelrc = np.ones(((initial_block_buffer * 2) + 1, (initial_block_buffer * 2) + 1))
    ### copy to separate taken and illegal... illegal stays in block_mask_in, used goes in block mask
    block_mask = block_mask_in.copy()
    block_mask[block_mask_in == 1] = 0
    print("- building blocks ", end="", flush=True)
    block_idx = 0
    while(block_idx < n_blocks):
        ### create narrow list by applying kernel to the mask
        buffer_block_mask = np.int16(convolve2d(block_mask, kernelrc, mode='same') > 0)
        ### now set all illegal points to off-limits (we don't need a buffer around them)
        buffer_block_mask += block_mask_in
        ### obtain a list of all locations where we can sample from (not taken, outside buffer, not illegal)
        narrowlist = np.where(buffer_block_mask == 0)
        ### if there are no samples we have a problem....
        if len(narrowlist[0]) == 0:
            print("x--> failed to generate blocks with buffer =", initial_block_buffer)
            ### restart with smaller buffer?
        else:
            ### randomly select a start point from the narrow list
            rand_idx = int(np.random.uniform(0, len(narrowlist[0])))
            rand_ctr_i = narrowlist[0][rand_idx]
            rand_ctr_j = narrowlist[1][rand_idx]
            ### verify that centerpoint is legal...
            if block_mask[rand_ctr_i, rand_ctr_j] == 0 and block_mask_in[rand_ctr_i, rand_ctr_j] == 0:

                if block_idx % (n_blocks // 5) == 0:
                    print("-", end="", flush=True)
                ### number of samples to add
                to_add = min(n_samples - (block_idx*approx_each), approx_each)
                ### use a cap to prevent infinite loops...
                iter_radius = -1
                ### random number, either 1 or -1
                iter_direction_i = (int(np.random.uniform(0, 2)) * 2) - 1
                iter_direction_j = (int(np.random.uniform(0, 2)) * 2) - 1
                ### iterate around in circles adding samples to block until we have enough
                while to_add > 0 and iter_radius < cost_cap:
                    ### increase radius by 2... next ring out
                    iter_radius += 2
                    for i in range(iter_radius):
                        for j in range(iter_radius):
                            ### determine next point to attempt to add to block
                            temp_i = rand_ctr_i - (iter_direction_i * (iter_radius // 2)) + (iter_direction_i * i)
                            temp_j = rand_ctr_j - (iter_direction_j * (iter_radius // 2)) + (iter_direction_j * j)
                            ### verify the point is in-bounds
                            if temp_i >= 0 and temp_i < block_mask_in.shape[0] and \
                                temp_j >= 0 and temp_j < block_mask_in.shape[1]:
                                if block_mask_in[temp_i, temp_j] == 0 and block_mask[temp_i, temp_j] == 0:
                                    to_add -= 1
                                    block_mask[temp_i, temp_j] = 2
                                    if to_add <= 0:
                                        break
                        if to_add <= 0:
                            break
                if iter_radius >= cost_cap:
                    print(" -s-> cost cap (size of iterative block) exceeded:", iter_radius)
                    block_idx -= 1
            ### if we pick a non-legal centerpoint, try again
            else:
                print("o", end="", flush=True)
                block_idx -= 1
        block_idx += 1
    print("->| done")
    return block_mask + block_mask_in

def qbi_2(block_mask_in, n_blocks, n_samples, approx_each, cost_cap, approx_size, initial_block_buffer):
    ### copy to separate taken and illegal... illegal stays in block_mask_in, used goes in block mask
    block_mask = block_mask_in.copy()
    ### sample mask for lazily drawing over
    sample_mask = block_mask_in.copy()
    print("- building blocks (qbi2) (step=5) ", end="", flush=True)
    successes = 0
    block_idx = 0
    ### new strategy: set values within size+ibb of point in block_mask to 2
    while block_idx < n_blocks:
        ### compute narrow list of legal points
        narrowlist = np.where(sample_mask == 0)
        ### mask with...
        if len(narrowlist[0]) == 0:
            print("x--> failed to generate blocks with buffer =", initial_block_buffer, "trying with smaller buffer")
            ### restart with smaller buffer?
        else:
            ### randomly select a start point from the narrow list
            rand_idx = int(np.random.uniform(0, len(narrowlist[0])))
            rand_ctr_i = narrowlist[0][rand_idx]
            rand_ctr_j = narrowlist[1][rand_idx]
            if block_mask[rand_ctr_i, rand_ctr_j] == 0:
                ### we have a legal sample center
                if block_idx % (n_blocks // 20) == 0:
                    print("-", end="", flush=True)

                ### start with large block, and overlay
                ### determine how many 0s are within the region
                ### enlarge the block until we have enough zeros
                ### draw arange over the block
                ### remove samples from end until we get back to to_add samples

                ### number of samples to add
                to_add = min(n_samples - (block_idx * int(approx_each)), int(approx_each))
                ### initial size of search block
                search_size = approx_size
                ### random number, either 1 or -1
                iter_corner_i = int(np.random.uniform(0, 2))
                iter_corner_j = int(np.random.uniform(0, 2))
                iter_direction = (int(np.random.uniform(0, 2)) * 2) - 1
                while True:
                    ### create np array of size search_size
                    search_ul = (max(0, rand_ctr_i - search_size//2), max(0, rand_ctr_j - search_size//2))
                    search_lr = (min(block_mask.shape[0]-1, rand_ctr_i + search_size - search_size//2),
                                 min(block_mask.shape[1]-1, rand_ctr_j + search_size - search_size//2))
                    ### how many zero values do we have in this region
                    search_yield = np.where(block_mask[search_ul[0]:search_lr[0], search_ul[1]:search_lr[1]] == 0)
                    if len(search_yield[0]) >= to_add:
                        ### all good, we can now draw block
                        ### issue: determine which edges to drop
                        ### idea: randomly pick a corner and direction, revert original values
                        full_block_search = (search_yield[0] + search_ul[0], search_yield[1] + search_ul[1])
                        ### set to iter value
                        block_mask[full_block_search] = block_idx + 4

                        ### now backtrack until we have the right number in total...
                        over_amt = len(search_yield[0]) - to_add
                        if over_amt > 0:
                            ### pick a corner by picking an i and a j
                            over_corner = ([search_ul[0], search_lr[0]][iter_corner_i],
                                           [search_ul[1], search_lr[1]][iter_corner_j])
                            iter_dir_i = (iter_corner_i * -2) + 1
                            iter_dir_j = (iter_corner_j * -2) + 1
                            ### pick an iteration direction
                            if iter_direction == 1:
                                for over_i in range(search_lr[0] - search_ul[0]):
                                    for over_j in range(search_lr[1] - search_ul[1]):
                                        i = over_corner[0] + (over_i * iter_dir_i)
                                        j = over_corner[1] + (over_j * iter_dir_j)
                                        if block_mask[i, j] == block_idx + 4:
                                            block_mask[i, j] = 0
                                            over_amt -= 1
                                            if over_amt <= 0:
                                                break
                                    if over_amt <= 0:
                                        break
                            else:
                                for over_j in range(search_lr[1] - search_ul[1]):
                                    for over_i in range(search_lr[0] - search_ul[0]):
                                        i = over_corner[0] + (over_i * iter_dir_i)
                                        j = over_corner[1] + (over_j * iter_dir_j)
                                        if block_mask[i, j] == block_idx + 4:
                                            block_mask[i, j] = 0
                                            over_amt -= 1
                                            if over_amt <= 0:
                                                break
                                    if over_amt <= 0:
                                        break
                        ### now mask out selection in sample_mask
                        mask_ul = (max(0, search_ul[0]-initial_block_buffer), max(0, search_ul[1]-initial_block_buffer))
                        mask_lr = (min(block_mask.shape[0]-1, search_lr[0] + initial_block_buffer),
                                     min(block_mask.shape[1]-1, search_lr[1] + initial_block_buffer))
                        sample_mask[mask_ul[0]: mask_lr[0], mask_ul[1]: mask_lr[1]] = 3
                        successes += 1
                        break
                    else:
                        ### not enough samples in grid... need to increase size
                        search_size += 2
                        if search_size > cost_cap:
                            print("-s-> cost cap (size of iterative block) exceeded:", search_size)
                            print(" - center:", (rand_ctr_i, rand_ctr_j), " search_ul/lr", search_ul, search_lr)
                            print(" - rand:", rand_idx, "out of", len(narrowlist[0]), " ... ", )
                            break

            else:
                print("o", end="", flush=True)
                block_idx -= 1
        block_idx += 1
    print("->| done")
    print("-", successes, "successful")
    return block_mask
def quick_block_split(legal_sample_idx_list, partition, split_blocks_nregions, guide_shape, split_blocks_buffer,
                      fold_name, layer_proj, y_base, n_splits, split_outer_buffer, sample_res_factor):
    val_fold_indices = []
    train_fold_indices = []
    n_total_samples = len(legal_sample_idx_list)
    n_test_samples = int(n_total_samples * partition[0])
    n_val_samples = int((n_total_samples - n_test_samples) * partition[1])
    ### approximate number of samples per block (test, val)
    approx_each = (n_test_samples / split_blocks_nregions, n_val_samples / split_blocks_nregions)
    ### approximate size (side length) of blocks (test, val)
    approx_block_size = (math.ceil(math.sqrt(approx_each[0])), math.ceil(math.sqrt(approx_each[1])))

    print("total samples:", n_total_samples, "test partition:", n_test_samples, "val partition:", n_val_samples)
    print("n blocks:", split_blocks_nregions, "approximate samples per block:", approx_each,
          "approximate block sizes:", approx_block_size)

    ###
    block_full_crs = (layer_proj[y_base][0][0], layer_proj[y_base][0][1] * sample_res_factor,
                                layer_proj[y_base][0][2], layer_proj[y_base][0][3], layer_proj[y_base][0][4],
                                layer_proj[y_base][0][5] * sample_res_factor)

    ### set up block mask
    block_mask = np.ones(guide_shape, dtype=np.int16)
    ### set up array for metaindices
    metaindex_arr = np.zeros(guide_shape)
    for i in range(len(legal_sample_idx_list)):
        ti, tj = legal_sample_idx_list[i]
        block_mask[ti, tj] = 0
        metaindex_arr[ti, tj] = i
    print("set up block mask (step 1)")

    ### diagnostic... make tifs for visual confirmation
    os.system("rm " + fold_name + "/fold_box_geotifs/*")
    os.system("mkdir " + fold_name + "/fold_box_geotifs")
    raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "diagnostic_initial_mask", block_mask, block_full_crs,
                               layer_proj[y_base][1], -1)
    print("- saved diagnostic initial geotif visualizations")

    ### first, do test...
    ### qbi_2(block_mask_in, n_blocks, n_samples, approx_each, cost_cap, approx_size, initial_block_buffer)
    block_mask = qbi_2(block_mask, split_blocks_nregions, n_test_samples, approx_each[0], min(guide_shape)//4,
                       approx_block_size[0], split_blocks_buffer)
    print("performed test partition (step 2)")
    block_mask[block_mask >= 2] = 2
    test_indices = metaindex_arr[np.where(block_mask == 2)]
    remaining_indices = metaindex_arr[np.where(block_mask == 0)]
    print("- found test indices:", len(test_indices), "out of", n_test_samples, "desired")
    ### make tifs for visual confirmation
    #dimension_override / base_res[y_base]
    raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "test_extent", block_mask, block_full_crs,
                               layer_proj[y_base][1], -1)

    print("- saved test geotif visualizations")
    ### now reset block mask for validation
    block_mask[block_mask != 0] = 1

    ### now do xval-splits
    for split_i in range(n_splits):
        print("- fold", split_i)
        split_mask_i = qbi_2(block_mask, split_blocks_nregions, n_test_samples, approx_each[1], min(guide_shape) // 4,
                           approx_block_size[1], split_blocks_buffer)

        val_fold_indices.append(metaindex_arr[np.where(split_mask_i >= 2)])
        train_fold_indices.append(metaindex_arr[np.where(split_mask_i == 0)])
        np.savetxt(fold_name + "/train_" + str(split_i) + ".csv", train_fold_indices[-1], delimiter=",")
        np.savetxt(fold_name + "/val_" + str(split_i) + ".csv", val_fold_indices[-1], delimiter=",")

        ### make geotif for visual inspection
        raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "val_extent_" + str(split_i), split_mask_i,
                                   block_full_crs, layer_proj[y_base][1], -1)
        print("- saved fold", split_i, "train/val geotif visualizations")
    print("- found training and validation indices")
    print("- saved training and validation indices")
    print("- saved training and validation geotif visualizations")
    np.savetxt(fold_name + "/test.csv", test_indices, delimiter=",")
    np.savetxt(fold_name + "/remaining.csv", remaining_indices, delimiter=",")
    print("- saved test indices")

    return test_indices, remaining_indices, train_fold_indices, val_fold_indices

def load_splits(fold_name, n_splits):
    test_indices = None
    remaining_indices = None
    train_fold_indices = []
    val_fold_indices = []
    test_indices = np.genfromtxt(fold_name + "/test.csv", delimiter=",")
    remaining_indices = np.genfromtxt(fold_name + "/remaining.csv", delimiter=",")
    for i in range(n_splits):
        train_fold_indices.append(np.genfromtxt(fold_name + "/train_" + str(i) + ".csv", delimiter=","))
        val_fold_indices.append(np.genfromtxt(fold_name + "/val_" + str(i) + ".csv", delimiter=","))
    return test_indices, remaining_indices, train_fold_indices, val_fold_indices

def fullrand_split(legal_sample_idx_list, partition, n_splits, fold_name, meta_indices):
    train_fold_indices = []
    val_fold_indices = []
    np.random.shuffle(meta_indices)
    test_indices = meta_indices[:int(len(meta_indices) * partition[0])]
    remaining_indices = meta_indices[int(len(meta_indices) * partition[0]):]
    np.random.shuffle(remaining_indices)
    print("- found test indices")
    for i in range(n_splits):
        np.random.shuffle(remaining_indices)
        val_fold_indices.append(remaining_indices[:int(len(remaining_indices) * partition[1])])
        train_fold_indices.append(remaining_indices[int(len(remaining_indices) * partition[1]):])
        np.savetxt(fold_name + "/train_" + str(i) + ".csv", train_fold_indices[-1], delimiter=",")
        np.savetxt(fold_name + "/val_" + str(i) + ".csv", val_fold_indices[-1], delimiter=",")
    print("- found training and validation indices")
    print("- saved training and validation indices")
    np.savetxt(fold_name + "/test.csv", test_indices, delimiter=",")
    np.savetxt(fold_name + "/remaining.csv", remaining_indices, delimiter=",")
    print("- saved test indices")

def make_pyramid_layer(pyparams):
    ### extract individual params from list
    fold_name, h5_chunk_size, expected_cube_size, dimension_override, legal_sample_idx_list, k, layer_crs, \
    test_indices, buffer_fill, n_splits, train_fold_indices, sample_min, sample_max, fold_min, fold_max, y_base, \
    sample_to_res, layer_data, buffer_dist, half_offset, center_offset = pyparams

    ### NOW MAKE PYRAMID LAYER
    ### remove any pre-existing layer files from directory
    os.system("rm " + fold_name + "/layer_" + str(k) + ".h5")
    ### create new h5 file for this layer -- these sizes should be consistent with dimension override
    h5_data_file_i = h5py.File(fold_name + "/layer_" + str(k) + ".h5", "a")
    h5_file_set_i = h5_data_file_i.create_dataset("data", (h5_chunk_size, sample_to_res, sample_to_res),
                                        maxshape=(None, sample_to_res, sample_to_res),
                                        chunks=(h5_chunk_size, sample_to_res, sample_to_res))

    ### initialize np array for h5 chunk
    h5_chunk_i = np.zeros((h5_chunk_size, sample_to_res, sample_to_res))
    h5_chunk_i.fill(-1)

    h5_chunk_counter = 0
    ### use roundup 2a if the layer is already at the desired output resolution
    ### this can be way more efficient becasue we don't need to draw sampling grid etc, can just extract from centers of
    ### pixels
    if sample_to_res == expected_cube_size:
        roundup_active = roundup_layer_2
        print("-", k, "running roundup 2")
    ### use roundup 3a if we need to sample at a non-native resolution
    ### this requires drawing grid at sampling resolution, more costly
    else:
        roundup_active = roundup_layer_3
        print("- running roundup 3")

    ### deal with sampling dims...?
    ### iterate over legal samples
    for i in range(len(legal_sample_idx_list)):
        ### roundup sample at this location
        result = roundup_active(k, legal_sample_idx_list[i], layer_crs, y_base, layer_data, expected_cube_size,
                                 buffer_dist, half_offset, center_offset, sample_to_res)
        ### add to h5 chunk
        h5_chunk_i[h5_chunk_counter % h5_chunk_size, :, :] = result
        ### if this sample is train/val, ... collect min/max info for later scaling
        if i not in test_indices:
            ### get scaling info
            mmcheck = result[result != buffer_fill]
            if len(mmcheck > 0):
                nanmin = np.nanmin(mmcheck)
                nanmax = np.nanmax(mmcheck)
                ### adjust min/max for combined data
                sample_min = min(nanmin, sample_min)
                sample_max = max(nanmax, sample_max)
                ### adjust min/max for individual splits
                for j in range(n_splits):
                    if i in train_fold_indices[j]:
                        fold_min[j] = min(nanmin, fold_min[j])
                        fold_max[j] = max(nanmax, fold_max[j])

        ### increment number of samples added to chunk. When the chunk is full,
        ### move data to the h5 file, clear, and refill chunk.... because adding to h5 has overhead
        h5_chunk_counter += 1
        if h5_chunk_counter % h5_chunk_size == 0:
            ### resize to current number of items
            h5_file_set_i.resize(h5_chunk_counter, axis=0)
            ### set values from current chunk
            h5_file_set_i[h5_chunk_counter - h5_chunk_size:h5_chunk_counter, :, :] = \
                np.array(h5_chunk_i[:, :, :])
            h5_chunk_i = np.zeros(h5_chunk_i.shape)
            h5_chunk_i.fill(-1)

    ### when we are out of samples, move remaining samples in chunk to h5
    ### then save h5 file
    h5_file_set_i.resize(h5_chunk_counter, axis=0)
    h5_file_set_i[h5_chunk_counter - (h5_chunk_counter % h5_chunk_size):h5_chunk_counter, :, :] = \
        np.array(h5_chunk_i[:h5_chunk_counter % h5_chunk_size, :, :])
    h5_data_file_i.close()

    return sample_min, sample_max, fold_min, fold_max


### TODO -- what does this do
def make_pyramids_main(cube_res, fold_name, h5_chunk_size, expected_cube_size, dimension_override,
                       legal_sample_idx_list, layer_crs, y_base, test_indices, buffer_fill, n_splits, layer_data,
                       train_fold_indices, sample_to_res, buffer_dist, half_offset, center_offset, parallel, lowmem):

    print("- set up h5 datasets")

    ### TODO -- NEED TO ACCOUNT FOR BUFFER DIST
    ### TODO -- KEEP IN MIND THAT LEGAL SAMPLES ARE IN BUFFER-FREE CRS
    ### setup work
    ### for folds, organized as [fold][layer]
    samples_mins = []
    samples_maxs = []
    folds_mins = []
    folds_maxs = []
    for i in range(len(layer_data)):
        folds_mins.append([])
        folds_maxs.append([])
        for j in range(n_splits):
            folds_mins[i].append(float("inf"))
            folds_maxs[i].append(float("-inf"))

    for i in range(len(layer_data)):
        samples_mins.append(float("inf"))
        samples_maxs.append(float("-inf"))

    if lowmem:
        ### we reloaded the layers without buffer so... no need to do anything with buffer distances..!!!
        buffer_dist = [0 for ii in range(len(buffer_dist))]

    ### set up parameters to pass cleanly to make_pyramid_layer in parallel (just reformatting arguments above)
    pyramid_params = []
    for i in range(len(cube_res)):
        pyramid_params.append([fold_name, h5_chunk_size, expected_cube_size[i], dimension_override,
                               legal_sample_idx_list, i, layer_crs, test_indices, buffer_fill, n_splits,
                               train_fold_indices, samples_mins[i], samples_maxs[i], folds_mins[i], folds_maxs[i],
                               y_base, sample_to_res[i], layer_data[i], buffer_dist[i], half_offset[i],
                               center_offset[i]])

    ### map arguments to make_pyramid_layer function and collect results
    if parallel:
        with Pool(None) as mpool:
            resarr = mpool.map(make_pyramid_layer, pyramid_params)
    else:
        resarr = []
        for i in range(pyramid_params):
            resarr.append(make_pyramid_layer(pyramid_params))
        print("- caution: running pyramids_main in nonparallel mode")


    print("- reformatted data to h5")
    print("- saved h5 files")
    print("- calculated min/max normalization parameters")
    ### do some work with min/max info we collected...
    for i in range(len(resarr)):
        sample_min, sample_max, fold_min, fold_max = resarr[i]
        folds_mins[i] = fold_min
        folds_maxs[i] = fold_max
        samples_mins[i] = sample_min
        samples_maxs[i] = sample_max

    combined_mins = np.array(samples_mins)
    combined_maxs = np.array(samples_maxs)
    np.savetxt(fold_name + "/norm_layer_mins_combined.csv", combined_mins, delimiter=",")
    np.savetxt(fold_name + "/norm_layer_maxs_combined.csv", combined_maxs, delimiter=",")

    fold_np_mins = np.array(folds_mins).transpose()
    fold_np_maxs = np.array(folds_maxs).transpose()
    for j in range(n_splits):
        np.savetxt(fold_name + "/norm_layer_mins_fold_" + str(j) + ".csv", fold_np_mins[j], delimiter=",")
        np.savetxt(fold_name + "/norm_layer_maxs_fold_" + str(j) + ".csv", fold_np_maxs[j], delimiter=",")
    print("- saved min/max normalization parameters")
    print("- done")


### should always return a 2d np array... previously roundup_layer_2a
def roundup_layer_2(k, base_idx, crs_list, yloc, layer_data, expected_cube_size, buffer_dist, half_offset,
                    center_offset, sample_to_res):
    ### this is legal sample id without buffer offset
    bi, bj = base_idx
    ### so this operation is ok (no buffer offset)
    ### getting the geographic center of this grid square...
    geo_ctr = idx_geo(bi + 0.5, bj + 0.5, crs_list[yloc])
    if k == yloc and expected_cube_size == 1:
        ### account for buffer offset in layer file
        return layer_data[[[bi+buffer_dist]], [[bj+buffer_dist]]]
    elif expected_cube_size == 1:
        ### this is again ok because geo_ctr is without buffer offset
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        ### account for buffer offset in layer file here
        return layer_data[[[int(tidi)+buffer_dist]], [[int(tidj)+buffer_dist]]]
    else:
        ### need to determine UL
        ### this is ok because geo_ctr is without buffer offset
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        sulx = int(tidi + half_offset) - center_offset
        suly = int(tidj + half_offset) - center_offset
        ### account for buffer offset here
        return layer_data[sulx+buffer_dist: sulx+buffer_dist+expected_cube_size,
                          suly+buffer_dist: suly+buffer_dist+expected_cube_size]

### should always return a 2d np array... previously roundup_layer_3a
def roundup_layer_3(k, base_idx, crs_list, yloc, layer_data, expected_cube_size, buffer_dist, half_offset,
                    center_offset, sample_to_res):
    ### this is legal sample id without buffer offset
    bi, bj = base_idx
    ### so this equation is ok (no buffer)
    geo_ctr = idx_geo(bi + 0.5, bj + 0.5, crs_list[yloc])
    if k == yloc and expected_cube_size == 1:
        ### account for buffer offset in layer file
        return layer_data[[[bi+buffer_dist]], [[bj+buffer_dist]]]
    elif expected_cube_size == 1:
        ### this is again ok because geo_ctr is without buffer offset
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        ### account for buffer offset in layer file here
        return layer_data[[[int(tidi)+buffer_dist]], [[int(tidj)+buffer_dist]]]
    else:
        ### need to determine UL
        ### this is ok because geo_ctr is without buffer offset
        tidi, tidj = geo_idx(geo_ctr[0], geo_ctr[1], crs_list[k])
        result = np.zeros((sample_to_res, sample_to_res))
        ### TODO -- rethink?
        ### account for buffer offset here, before resampling this time
        sulx = int(tidi + half_offset - 0.5) + buffer_dist - center_offset
        suly = int(tidj + half_offset - 0.5) + buffer_dist - center_offset
        iterstep = (2 * (center_offset - half_offset) + 1) / (sample_to_res - 1)
        for i in range(sample_to_res):
            for j in range(sample_to_res):
                result[i, j] = layer_data[int(sulx + i*iterstep), int(suly + j*iterstep)]
        return result

def save_info_file(data_info, expected_sample_to, fold_name, n_splits, buffer_fill, data_input_crs,
                   np_random_seed):
    ### SAVE INFO
    with open(fold_name + "/info.txt", 'w') as infofile:
        infofile.write(str(n_splits) + "," + str(buffer_fill) + "," + data_input_crs +
                       "," + str(np_random_seed))
        for i in range(len(data_info)):
            infofile.write("\n" + str(expected_sample_to[i]) + "," + str(data_info[i]["xy"]) + "," +
                           str(data_info[i]["index"]) + "," + str(data_info[i]["name"]))
    print("- saved info file")

def set_checkpoint(fold_name, checkpoint_number):
    with open(fold_name + "/checkpoint.txt", 'w') as checkpoint:
        checkpoint.write(str(checkpoint_number))
    print("- checkpoint:", checkpoint_number)

def get_checkpoint(fold_name):
    with open(fold_name + "/checkpoint.txt", 'r') as checkpoint:
        checkpoint_number = checkpoint.read()
    return int(checkpoint_number)