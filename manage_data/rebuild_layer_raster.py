import sys

sys.path.append("..")
import os
import numpy as np
from osgeo import gdal
from osgeo import osr
from utils import utils
import create_pyramid_functions as cpf
import h5py
import raster_helpers

### idea -- load the guiding raster layer and various parameters from config file
### iterate over h5 samples
### try to rebuild them into geotifs
gdal.UseExceptions()

### DEAL WITH PARAMETERS FROM CONFIG FILE...
config_dir = "../manage_data/configs/"
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

### representative layers ... one from each layer res group
repr_layers = [0, y_base]


### now... load h5 files
layer_locs = []
h5_src = []
h5_data = []
print("init layer h5s")
for k in range(len(data_info)):
    layer_locs.append(fold_name + "/layer_" + str(k) + ".h5")
    h5_src.append(h5py.File(layer_locs[k],'r'))
    h5_data.append(h5_src[k]["data"])

test_index = np.genfromtxt(fold_name + "/test.csv", delimiter=',').astype(int)
combined_index = np.genfromtxt(fold_name + "/remaining.csv", delimiter=',').astype(int)

print("loading guiding layer shape")
guiding_layer_raster = gdal.Open(data_info[y_base]["loc"])
guiding_grid_size = (guiding_layer_raster.RasterXSize, guiding_layer_raster.RasterYSize)
guiding_geo = guiding_layer_raster.GetGeoTransform()
guiding_prj = guiding_layer_raster.GetProjection()
print("init guiding grid np")
guiding_block = np.zeros(guiding_grid_size) - 1
#metaindex_arr = np.zeros(guiding_grid_size)

rebuild_indices = []
### issue: need to reverse engineer raster coords from index...
legal_sample_idx_list = cpf.load_legal_sample_ids(fold_name)

print("extracting test sample values (guiding grid)")
for i in range(len(test_index)):
    position_value_np = h5_data[y_base][test_index[i], :, :]
    #print(position_value_np.shape)
    #print(position_value_np)
    #print(type(legal_sample_idx_list[test_index[i]]))
    #print(legal_sample_idx_list[test_index[i]])
    guiding_block[legal_sample_idx_list[test_index[i]][0], legal_sample_idx_list[test_index[i]][1]] = position_value_np


print("extracting train/val sample values (guiding grid)")
for j in range(len(combined_index)):
    position_value_np = h5_data[y_base][combined_index[j], :, :]
    guiding_block[legal_sample_idx_list[combined_index[j]][0], legal_sample_idx_list[combined_index[j]][1]] = position_value_np
    #print(position_value_np.shape)
    #p#rint(position_value_np)
    #p#rint(legal_sample_idx_list[combined_index[j]])


print("done extracting sample values from h5")
### reformat into geotif and save
### first set up output directory
os.system("rm ../data/diagnostic/layer_reconstruction")
os.system("mkdir ../data/diagnostic")
os.system("mkdir ../data/diagnostic/layer_reconstruction")
#layer_raster.GetGeoTransform(), layer_raster.GetProjection()
raster_helpers.save_raster(fold_name + "/fold_box_geotifs", "diagnostic_initial_mask", guiding_block, guiding_geo,
                               guiding_prj, -1)
