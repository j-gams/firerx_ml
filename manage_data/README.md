## Manage Data -- Overview
The manage_data directory contains tools for manageing data, particularly for cleaning raw raster data, aligning raster data for use in causal inference, and extracting analysis-ready datasets suitable for machine learning from raw raster data (both data cubes and data pyramids).

See more about the functionality of specific files, or how to align data for causal inference or build analysis-ready datasets:\
[Files](#files)\
[Causal Inference Alignment](#ci)\
[Build Analysis-Ready Datasets](#mldata)\
[Appendix](#appendix)


### 1. Files <a name="files"></a>
 - batch_align_raster.py ([description](#batch_align))
 - align_raster.py ([description](#align_raster))
 - align_ci_helpers.py ([description](#align_cih))
 - trim_reproj_rasters.py ([description](#trim_reproj))
 - create_pyramid_set.py ([description](#pyramid))
 - create_pyramid_functions.py ([description](#pyramid_func))
 - raster_helpers.py ([description](#pyramid_help))
 - aws_check.sh ([description](#aws_check))
 - aws_psychic.sh ([description](#aws_psychic))
 - configs ([description](#configs))

#### 1.1 batch_align_raster.py <a name="batch_align"></a>
This is the new and improved main file for cleaning raw raster files and aligning them for Causal Inference. 
This process involves trimming the raster layers to the area of interest, reprojecting the raster layers to a consistent projection, and then resampling layers to have a consistent CRS and resolution. 
This updaet contains several optimizations over align_raster.py, and allows for the automatic slicing and batching of larger states or regions, allowing for better scaling.
This update also contains user improvements, particularly the streamlining of parameters.
The default config for this process is dbci_default_config.json. The parameters are as follows:

| Parameter                        | Values | Function                                                                                                                                |
|----------------------------------| --- |-----------------------------------------------------------------------------------------------------------------------------------------|
| core:multiprocessing             | True/False | True is recommended. This parameter determines whether alignment of layers will be parallelized, which can result in huge time savings. |
| name:collection_name             |        |                                                                                                                                         |
| name:collection_directory_prefix |        |                                                                                                                                         |
| name:trim_step_out               |        |                                                                                                                                         |
| name:reproj_step_out             |        |                                                                                                                                         |
| name:align_step_out              |        |                                                                                                                                         |
| name:raster_src                  |        |                                                                                                                                         |
| params:subdiv_meter_overlap      |        |                                                                                                                                         |
| params:subdiv_meter_slop         |        |                                                                                                                                         |
| params:subdiv_hv_split           |        |                                                                                                                                         |
| params:res_check_tol             |        |                                                                                                                                         |
| params:output_nodata             |        |                                                                                                                                         |
| params:resampling_mode           |        |                                                                                                                                         |
| skip:guiding_load                |        |                                                                                                                                         |
| skip:subdivision                 |        |                                                                                                                                         |
| skip:trim                        |        |                                                                                                                                         |
| skip:extent_reproj               |        |                                                                                                                                         |
| skip:raster_reproj               |        |                                                                                                                                         |
| skip:resolution_check            |        |                                                                                                                                         |
| skip:alignment                   |        |                                                                                                                                         |
| data:guiding_layer_idx           |        |                                                                                                                                         |
| data:extent_epsg_override        |        |                                                                                                                                         |
| data:extent_info                 |        |                                                                                                                                         |
| data:data_info                   |        |                                                                                                                                         |
| data:exclude                     |        |                                                                                                                                         |

#### 1.2 align_raster.py <a name="align_raster"></a>
This is the old and busted file for cleaning raw raster files and aligning them for Causal Inference. Use at your own peril. 
This process involves trimming the raster layers to the area of interest, reprojecting the raster layers to a consistent projection, and then resampling layers to have a consistent CRS and resolution. 
Unlike batch_align_raster.py, this file contains a number of tools for working with AWS, including the ability to parallelize the alignment step over multiple AWS EC2 instances, and the ability to move files to S3 upon completion.
This file is clunkier and lacks other optimizations, plus the ability to slice and batch the alignment process.
The default config for this process is ci_default_config.json. The parameters are provided in the appendix.

#### 1.3 align_ci_helpers.py <a name="align_cih"></a>
#### 1.4 trim_reproject_rasters.py <a name="trim_reproj"></a>
#### 1.5 create_pyramid_set.py <a name="pyramid"></a>
This is the main file for building analysis ready datasets, from data pyramids to data cubes. The default config file for this process is mldata_default_config.json. The parameters are as follows:

| Parameter                     | Values                  | Function                                                                                                                                                                                                                     |
|-------------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| core:mode                     | (pyramid, cube, adjust) | This parameter dictates whether the program creates a set of data pyramids (pyramid), full-resolution data cubes (cube), or adjusted resolution data cubes (adjust)                                                          |
| core:run_name_suffix_default  | (string)                | The "name" of each run is dictated by the mode followed by this suffix (eg pyramid_set1 where the suffix is "set1"). This is used for directory names, etc.                                                                  |                                                                                                                                                                 |
| core: run_name_prefix         | (file path)             | The directory in which to create and save the resulting dataset. This can be relative to the manage_data directory. The recommended path is `../data/ml_sets/`       .                                                       |
 | params:dimension_override     | sample meters (int)     | This value dictates the size of samples to be built, eg when set to 1000 the program extracts samples on a 1000m*1000m grid. When set to -1, the size is determined by the grid size of the layer specified by `base_layer`. | 
 | params:cube_reduce_dim_factor | factor (scalar > 0)     | This value dictates the ratio of                                                                                                                                                                                             | 
#### 1.6 create_pyramid_functions.py <a name="pyramid_func"></a>
#### 1.7 raster_helpers.py <a name="pyramid_help"></a>
#### 1.8 aws_check.sh <a name="aws_check"></a>
#### 1.9 aws_psychic.sh <a name="aws_psychic"></a>

### 2. Align Data for Causal Inference <a name="ci"></a>
This is an overview of how to align a dataset for Causal Inference.
update the 
=======
See more about the functionality of specific files, or how to align data for causal inference or build analysis-ready datasets:
[Files](#files)
[Causal Inference Alignment](#ci)
[Build Analysis-Ready Datasets](#mldata)

### 1. Files <a name="files"></a>
 - align_raster.py
 - align_ci_helpers.py
 - create_pyramid_set.py
 - create_pyramid_functions.py
 - raster_helpers.py
 - configs
 - aws_check.sh
 - aws_psychic.sh
 - 
### 2. Align Data for Causal Inference <a name="ci"></a>
### 3. Build Analysis-Ready Datasets <a name="mldata"></a>
### 4. Appendix <a name="appendix"></a>

#### 4.1 align_raster.py parameters
Note that users are encouraged to use batch_align_raster.py instead of align_raster.py

| Parameter                         | Values                                  | Function                                                                                                                                                                                         |
|-----------------------------------|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| core:multiprocessing_mode         | True/False                              | True is recommended. This parameter determines whether alignment will be parallelized, which can result in huge time savings.                                                                    |
 | core:work_mode                    | (local, aws_main, aws_work, aws_bigbox) | ...                                                                                                                                                                                              |
 | core:aws_main_mode                | True/False                              | ...                                                                                                                                                                                              |
 | core:aws_work_mode                | True/False                              | ...                                                                                                                                                                                              |
 | core:aws_pems_dir                 | (file path)                             | ...                                                                                                                                                                                              |
 | core:aws_workers                  | (remote addresses)                      | ...                                                                                                                                                                                              |
 | core:aws_primary_box              | (integer)                               | ...                                                                                                                                                                                              |
 | core:aws_inst_directory           | (directory name)                        | ...                                                                                                                                                                                              |
 | core:aws_env_name                 | (conda env name)                        | ...                                                                                                                                                                                              |
| core:s3_aligned_loc               | (s3 address)                            | ...                                                                                                                                                                                              |
| core:s3_reproj_loc                | (s3 address)                            | ...                                                                                                                                                                                              |
| local:data_directory_prefix       | (file path)                             | The base directory for data, and the directory in which to create and save the aligned dataset. This can be relative to the manage_data directory. The recommended path is `../data/`            |
| local:raster_src                  | (file path)                             | The directory in which raw raster data is stored. The recommended path is `../data/raw_raster`                                                                                                   |
| local:extent_src                  | (file path)                             | The directory in which the area of interest extent is stored. The recommended path is `../data/extent/[REGION_NAME]`                                                                             |
| local:extent_name                 | (string)                                | The name of the extent to look for in the extent_src directory, minus any extensions (eg if the extent files include Colorado.prj, Colorado.shx, Colorado.shp, provide Colorado for extent_name) |
| local:extent_epsg_override        | (string)                                | Override the projection of the extent to this value when the given value is inaccurate/faulty                                                                                                    |
| local:auto_reduce_extent          | True/False                              |                                                                                                                                                                                                  |
| local:auto_reduce_extent_by       | <value>                                 |                                                                                                                                                                                                  |
| local:trim_step_out               | (directory name)                        | Directory to output trimmed raster files, to be created within the collection. The recommended directory name is `trimmed/`                                                                      |
| local:reproj_step_out             | (directory name)                        | Directory to output reprojected raster files, to be created within the collection. The recommended directory name is `reprojected/`                                                              |
| local:align_step_out              | (directory name)                        | Directory to output aligned raster files, to be created within the collection. The recommended directory name is `aligned/`                                                                      |
| local:collection_directory_prefix | (file path)                             | The base directory for the collection to be created. The recommended path is `../data/aligned_raster`                                                                                            |
| local:collection_name             | (directory name)                        | The name of this run is the directory in which all outputted files are saved (eg "align_colorado/")                                                                                              |
| local:raw_data_from_s3            | True/False                              | False is recommended, especially when working locally.                                                                                                                                           |
| local:check_local_s4              | True/False                              | False is recommended.                                                                                                                                                                            |
| local:raw_raster_s3_prefix        | (s3 address)                            | ...                                                                                                                                                                                              |
| local:raw_raster_s3_locs          | dict                                    | ...                                                                                                                                                                                              |
| local:raw_extent_s3_loc           | (s3 path)                               | ...                                                                                                                                                                                              |
| skip:guiding_load                 | True/False                              | Whether to skip loading the guiding layer first TODO                                                                                                                                             |
| skip:trim                         | True/False                              | Whether to skip the raster trim step                                                                                                                                                             |
| skip:raster_reproj                | True/False                              | Whether to skip the raster reprojection step                                                                                                                                                     |
| skip:extent_reproj                | True/False                              | Whether to skip the reprojection of the extent to match the projection of rasters being trimmed                                                                                                  |
| skip:resolution_check             | True/False                              | Whether to perform the check comparing the computed raster resolution to the given expected raster resolution                                                                                    |
| skip:raw                          | True/False                              | ...                                                                                                                                                                                              |
| skip:rm_existing_TEST             | True/False                              | Set to False. This variable is for testing/debugging only.                                                                                                                                       |
| skip:subset_data_TEST             | True/False                              | Set to False. This variable is for testing/debugging only.                                                                                                                                       |
| skip:worker_subset_TEST           | True/False                              | Set to False. This variable is for testing/debugging only.                                                                                                                                       |
| skip:skip_checkpoint              | int                                     |                                                                                                                                                                                                  |
| skip:aws_skip_fullsetup           | True/False                              |                                                                                                                                                                                                  |
| skip:aws_skip_transfer_raw        | True/False                              |                                                                                                                                                                                                  |
| skip:aws_skip_transfer            | list of indices                         |                                                                                                                                                                                                  |
| skip:aws_skip_begin               | list of indices                         |                                                                                                                                                                                                  |
| params:res_check_tol              | double                                  | 0.001 default value recommended. This value                                                                                                                                                      |
| params:output_nodata              | int                                     | -99999 default value recommended. This value                                                                                                                                                     |
| params:resampling_mode            | (n_meter,)                              | n_meter is currently the only working (and recommended) resampling mode                                                                                                                          |
| data:align_to_layer               | dict                                    |                                                                                                                                                                                                  |
| data:data_info                    | list                                    |                                                                                                                                                                                                  |
| data:subset                       | list or None                            |                                                                                                                                                                                                  |
 
