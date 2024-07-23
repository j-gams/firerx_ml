### create_pyramid_set.py

### DO IMPORTS
### name == main necessary for multiprocessing
if __name__ == "__main__":
    print ("importing...")
    import sys
    sys.path.append("..")
    import os
    import psutil
    import time
    import numpy as np
    from osgeo import gdal
    gdal.UseExceptions()
    #import raster_helpers ### TODO -- what is this
    import create_pyramid_functions as cpf
    from utils import utils
    starttime = time.process_time()

    ### DEAL WITH PARAMETERS FROM CONFIG FILE...
    config_dir = "configs/"
    config_prefix = "mldata_"
    config_name = "default_config"
    if len(sys.argv) > 2 and sys.argv[2][:6] == "config":
        config_name = sys.argv[2][7:]
    config_loc = config_dir + config_prefix + config_name + ".json"
    ### load config
    config = utils.read_config(config_loc)
    ### distribute config values to parameters...
    print("loaded config file")

    ### CORE PARAMETERS
    ### whether to run in pyramid, cube, or adjusted cube mode
    data_mode = config["core"]["mode"]
    ### default name for this run
    run_name_suffix = config["core"]["run_name_suffix_default"]
    ### fold name prefix
    run_name_prefix = config["core"]["run_name_prefix"]
    ### combine to create run name and path
    run_name = data_mode + run_name_suffix
    fold_name = run_name_prefix + run_name
    ### whether to run in low memory mode, deleting some large np arrays while doing train/val/test split
    low_memory_mode = config["core"]["low_memory_mode"]

    ### PROCESS PARAMETERS
    ### data dimension override -- create sample cubes/pyramids at this total resolution ### TODO
    dimension_override = config["params"]["dimension_override"]
    ### reduce cube sampling resolution to this fraction of total pyramid parameters when in adjusted cube mode
    cube_reduce_dim_factor = config["params"]["cube_reduce_dim_factor"]
    ### whether to work from checkpoints
    load_checkpoint = config["params"]["load_checkpoint"]
    ### data target projection --- just prints warnings if this doesn't match ### TODO
    data_target_proj = config["params"]["data_target_proj"]
    ### split method for geographic train/val/test partitioning
    split_method = config["params"]["split_method"]
    ### buffer distance for blocks split method
    split_blocks_buffer = config["params"]["split_blocks_buffer"]
    ### number of regions for blocks split method
    split_blocks_regions = config["params"]["split_blocks_regions"]
    ### outer buffer for split method
    split_outer_buffer = config["params"]["split_outer_buffer"]
    ### number of cross-validation splits
    partition_n_splits = config["params"]["partition_n_splits"]
    ### ratio to be set aside for testing and validation in partition
    partition = (config["params"]["partition_test_ratio"], config["params"]["partition_val_ratio"])
    ### exclude ### TODO
    exclude = config["params"]["exclude"]
    ### buffer fill
    buffer_fill = config["params"]["buffer_fill"]
    ### random seed
    np_random_seed = config["params"]["np_random_seed"]
    ### h5 chunk size -- number of samples to work on at once
    h5_chunk_size = config["params"]["h5_chunk_size"]
    ### whether to parallelize sample extraction with multiprocessing
    parallelize = config["params"]["parallelize"]

    ### retrieve data information from config file
    data_info = config["data"]["data_info"]
    ### subset data info for testing
    test_subset = config["data"]["test_subset"]
    if len(test_subset) > 0:
        data_info = [data_info[ii] for ii in test_subset]

    ### ALIGNMENT PARAM
    ### base layer -- layer to align around
    y_base = config["params"]["base_layer"] % len(data_info)


    ### USER PARAMETERS
    ### Required CRS: EPSG 4326 WSG 84
    data_input_crs = "EPSG4326WSG84" ### TODO

    print('creating fold "' + run_name +'"')
    print('- aligning with respect to layer', y_base, data_info[y_base]["name"])
    print("- setting random seed to", np_random_seed)

    ### auto params
    np.random.seed(np_random_seed)
    layer_locs = []
    base_res = []
    sample_to_res = []
    y_layers = []
    x_layers = []
    layer_names = []
    layer_type = []

    min_x = dimension_override
    ### extract some information about data...
    for i in range(len(data_info)):
        if i not in exclude:
            layer_locs.append(data_info[i]["loc"])
            base_res.append(data_info[i]["base_res"])
            sample_to_res.append(data_info[i]["base_res"])
            layer_names.append(data_info[i]["name"])
            layer_type.append(data_info[i]["type"])
            if data_info[i]["xy"] == "x":
                x_layers.append(i)
                if base_res[-1] < min_x:
                    min_x = base_res[-1]
            else:
                y_layers.append(i)

    ### either use override pyramid/cube size or base off y_base
    ### use dimension_override from now on
    if dimension_override == -1:
        dimension_override = base_res[y_base]

    ### set sampling resolution to min meter dimension (max resolution)
    if data_mode == "cube":
        for idx in x_layers:
            sample_to_res[idx] = min_x

    checkpoint_prev = -1
    if load_checkpoint:
        try:
            checkpoint_prev = cpf.get_checkpoint(fold_name)
            print("- previous checkpoint found:", checkpoint_prev)
        except:
            print("- error loading checkpoint, setting checkpoint to 0")
            checkpoint_prev = 0

    ### checkpoint 0
    if checkpoint_prev <= 0:
        os.system("mkdir " + fold_name)
        cpf.set_checkpoint(fold_name, checkpoint_number=0)
        print("- created base directory")

    layer_nodata = []
    layer_size = []
    layer_crs = []
    layer_proj = []
    layer_data = []

    ### LOAD DATA
    ### need to do regardless of checkpoint
    ### import raster layers, get data and crs info
    for item in layer_locs:
        layer_names.append(item.split("/")[-1].split(".")[0])
        layer_raster = gdal.Open(item)
        rasterband = layer_raster.GetRasterBand(1)
        layer_nodata.append(rasterband.GetNoDataValue())
        layer_size.append((layer_raster.RasterXSize, layer_raster.RasterYSize))
        tulh, tpxh, _, tulv, _, tpxv = layer_raster.GetGeoTransform()
        tpxv = abs(tpxv)
        layer_crs.append((tulh, tulv, tpxh, tpxv))
        layer_proj.append([layer_raster.GetGeoTransform(), layer_raster.GetProjection()])
        layer_data.append(layer_raster.ReadAsArray().transpose())
        del rasterband
        del layer_raster

    bytesum = 0
    print("- loaded raster data (" + str(len(layer_data)) + " layers)")
    print("- ndvals:", layer_nodata)
    for i in range(len(layer_data)):
        bytesum += layer_data[i].nbytes
    print("- layer data memory costs (GB):", (bytesum / 1000000000))
    print("- total psutil process memory (GB):", psutil.virtual_memory()[3]/1000000000)

    ### ORCHESTRATE

    ### compute expected cube size (expected size of each layer in pyramid)
    ### need to compute regardless of checkpoint
    print("computing expected cube sizes and sampling sizes")
    expected_cube_size = cpf.compute_expected_cube_sizes(dimension_override, base_res)
    expected_sample_to = cpf.compute_expected_cube_sizes(dimension_override, sample_to_res)
    ### adjust sampling resolution of cube to approximately some factor of pyramid resolution?
    if data_mode == "adjust":
        print("adjusting expected sampling sizes")
        expected_sample_to = cpf.reduce_from_total_dims(expected_cube_size, expected_sample_to, x_layers, cube_reduce_dim_factor)
    ### save info file
    cpf.save_info_file(data_info, expected_sample_to, fold_name, partition_n_splits, buffer_fill, data_input_crs, np_random_seed)
    ### make a buffer around data layers to avoid going out of bounds... involves resizing data
    print("padding data to avoid out-of-bounds errors")
    buffer_dist, layer_data = cpf.make_buffer(buffer_fill, layer_data, base_res, dimension_override)
    ### compute sampling offsets for dealing with odd and even sizes in sample generation
    center_offset, half_offset = cpf.compute_offsets(expected_cube_size, layer_data)

    bytesum = 0
    for i in range(len(layer_data)):
        bytesum += layer_data[i].nbytes
    print("- layer data memory costs (GB):", (bytesum / 1000000000))
    print("- total psutil process memory (GB):", psutil.virtual_memory()[3] / 1000000000)

    ### roundup 1: compile list of legal samples
    ### this means samples within the AOI, in particular where we have data for every layer
    ### checkpoint 1
    if checkpoint_prev <= 1:
        cpf.set_checkpoint(fold_name, checkpoint_number=1)
        ### This works with buffers
        legal_sample_idx_list, guide_shape, sample_res_factor = cpf.compile_legal_samples(expected_cube_size,
                                                                                          layer_data, y_base, base_res,
                                                                                          dimension_override,
                                                                                          buffer_fill, layer_crs,
                                                                                          layer_nodata, buffer_dist,
                                                                                          half_offset, center_offset)
        ### save legal sample index list
        ### these indices are in the non-buffer grid
        cpf.save_legal_sample_ids(legal_sample_idx_list, fold_name)
    ### if already computed, load...
    else:
        ### TODO -- buffer
        legal_sample_idx_list = cpf.load_legal_sample_ids(fold_name)
        guide_shape = (int((layer_data[y_base].shape[0] - (2*buffer_dist[y_base])) * (base_res[y_base] / dimension_override)),
                       int((layer_data[y_base].shape[1] - (2*buffer_dist[y_base])) * (base_res[y_base] / dimension_override)))
        sample_res_factor = dimension_override / base_res[y_base]
        print("base layer shape", layer_data[y_base].shape)
        print("guiding shape", guide_shape)
        print("sample resolution factor (sample res/base res)", sample_res_factor)
    ### now do train/test/val splits
    ### checkpoint 2
    if checkpoint_prev <= 2:
        cpf.set_checkpoint(fold_name, checkpoint_number=2)
        ### index of indices to make these easier to work with for partitioning
        meta_indices = np.arange(len(legal_sample_idx_list))

        if low_memory_mode:
            print("low memory mode: removing layer data while performing partitions")
            layer_data.clear()
            print("- total psutil process memory (GB):", psutil.virtual_memory()[3] / 1000000000)
        ### TODO -- verify these methods are good with buffer
        ### TODO -- probably fine?
        if split_method == "blocks":
            test_indices, remaining_indices, train_fold_indices, val_fold_indices = cpf.quick_block_split(legal_sample_idx_list, partition, split_blocks_regions, guide_shape, split_blocks_buffer, fold_name, layer_proj, y_base, partition_n_splits, split_outer_buffer, sample_res_factor)
        elif split_method == "fullrandom":
            test_indices, remaining_indices, train_fold_indices, val_fold_indices = cpf.fullrand_split(legal_sample_idx_list, partition, partition_n_splits, fold_name)
        if low_memory_mode:
            print("low memory mode: reloading layer data")
            ### reload data (only layer data)
            for item in layer_locs:
                layer_names.append(item.split("/")[-1].split(".")[0])
                layer_raster = gdal.Open(item)
                layer_data.append(layer_raster.ReadAsArray().transpose())
                del rasterband
                del layer_raster
            print("- total psutil process memory (GB):", psutil.virtual_memory()[3] / 1000000000)
    else:
        ### load indices
        test_indices, remaining_indices, train_fold_indices, val_fold_indices = cpf.load_splits(fold_name, partition_n_splits)

    ### parallelize pyramid setup
    ### checkpoint 3
    sampletime = time.process_time()
    if checkpoint_prev <= 3:
        cpf.set_checkpoint(fold_name, checkpoint_number=3)
        ### extract samples in parallel, or now also in series...
        cpf.make_pyramids_main(base_res, fold_name, h5_chunk_size, expected_cube_size, dimension_override,
                           legal_sample_idx_list, layer_crs, y_base, test_indices, buffer_fill, partition_n_splits,
                           layer_data, train_fold_indices, expected_sample_to, buffer_dist, half_offset,
                           center_offset, parallelize)
        print("- total psutil process memory (GB):", psutil.virtual_memory()[3] / 1000000000)
    else:
        print("- all done without reaching checkpoint")

    ### checkpoint 4
    ### save all
    cpf.set_checkpoint(fold_name, checkpoint_number=4)
    endtime = time.process_time()
    print("sampling time:", endtime - sampletime)
    print("total process time:", endtime - starttime)


