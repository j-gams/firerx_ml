### make/read config files...
### import json
import sys
import json

def make_config(out_dict, nameloc):
    with open(nameloc, "w") as config_out:
        json.dump(out_dict, config_out)

def read_config(in_loc):
    config_in = open(in_loc)
    config_dict = json.load(config_in)
    return config_dict

### to remake this python utils.py make_default_configs
### cd Desktop/work/firerx_ml/manage_data; conda activate gdal; python align_raster.py

# box1 ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-44-237-87-156.us-west-2.compute.amazonaws.com
# box2 ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-54-69-87-217.us-west-2.compute.amazonaws.com
# box3 ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-52-37-24-253.us-west-2.compute.amazonaws.com
# box4 ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-44-231-201-214.us-west-2.compute.amazonaws.com
# box5 ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-34-216-191-56.us-west-2.compute.amazonaws.com

args = sys.argv
if len(args) <= 1:
    args.append(None)

if sys.argv[1] == "read_config":
    print("testing read_config")
    config = read_config("util_testbin/configout1.json")
    print(config)
    print("done reading config")

if sys.argv[1] == "make_dbci_config":
    if len(sys.argv) > 2:
        dbciconf_name = sys.argv[2]
    else:
        dbciconf_name = "default_config"
    dbciconf_loc = "../manage_data/configs/dbci_" + dbciconf_name + "_ml.json"
    print("writing default dbci alignment config to", dbciconf_loc)
    ### make default DBCI config
    dbciconf_dict = {"core": {"multiprocessing_mode": True},
                     "name": {"collection_name": "colorado_ml_lf22/",
                              "collection_directory_prefix": "../data/aligned_raster/",
                              "trim_step_out": "trimmed/",
                              "reproj_step_out": "reprojected/",
                              "align_step_out": "aligned/",
                              "raster_src": "../data/raw_raster/"},
                     "params": {"subdiv_meter_overlap": 1000,
                                "subdiv_meter_slop": 10,
                                "subdiv_hv_split": (1, 1),
                                "res_check_tol": 0.001,
                                "output_nodata": -99999,
                                "resampling_mode": "n_meter"},
                     "skip": {"guiding_load": False,
                              "subdivision": True,
                              "trim": False,
                              "extent_reproj": False,
                              "raster_reproj": False,
                              "resolution_check": False,
                              "alignment": True},
                     "data": {"guiding_layer_idx": 0,
                              "extent_epsg_override": False, ### need 3785 for california
                              "exclude": [],
                              "extent_info": {"extent_src": "../data/extent/colorado/",
                                              "extent_name": "Colorado_State_Boundary"},
                              "data_info": [{"loc": "colorado_wue_2022.tif", "name": "ECOSTRESS_WUE_22",
                                             "base_res": 70, "output_res": 70, "resample_res": 10,
                                             "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                             "override_input_proj": False},
                                            {"loc": "conus_age_1km_albers.tif", "name": "TREEAGE_mean",
                                              "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                              "base_res": 1000, "output_res": 70, "resample_res": 10,
                                              "override_input_proj": "5070"},#},

                                             {"loc": "forest_aboveground_carbon_flux_ED_GEDI.tif","name": "GEDI_AGB",
                                              "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                              "base_res": 1000, "output_res": 70, "resample_res": 10,
                                              "override_input_proj": False},

                                             {"loc": "PRISM_tmax_30yr_normal_800mM5_annual_bil.bil",
                                              "name": "PRISM_TEMPMAX_30_mean",
                                              "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                              "base_res": 800, "output_res": 70, "resample_res": 10,
                                              "override_input_proj": False},
                                             {"loc": "PRISM_tmean_30yr_normal_800mM5_annual_bil.bil",
                                              "name": "PRISM_TEMPMEAN_30_mean",
                                              "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 800, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                             {"loc": "PRISM_tmin_30yr_normal_800mM5_annual_bil.bil",
                                              "name": "PRISM_TEMPMIN_30_mean",
                                               "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 800, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                              {"loc": "PRISM_vpdmax_30yr_normal_800mM5_annual_bil.bil",
                                               "name": "PRISM_VAPORMAX_30_mean",
                                               "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 800, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                              {"loc": "PRISM_vpdmin_30yr_normal_800mM5_annual_bil.bil",
                                               "name": "PRISM_VAPORMIN_30_mean",
                                               "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 800, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                              {"loc": "PRISM_ppt_30yr_normal_800mM4_annual_bil.bil",
                                               "name": "PRISM_PRECIP_30_mean",
                                               "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 800, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},

                                              {"loc": "LC22_EVC_230.tif", "name": "LANDFIRE_EVC_22_mean",
                                               "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 30, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                              {"loc": "LC22_EVH_230.tif", "name": "LANDFIRE_EVH_22_mean",
                                               "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 30, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                              {"loc": "LC22_EVT_230.tif", "name": "LANDFIRE_EVT_22_mode",
                                               "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mode",
                                               "base_res": 30, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                              {"loc": "LC22_FVH_230.tif", "name": "LANDFIRE_FVH_22_mean",
                                               "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 30, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                              {"loc": "LC22_CC_230.tif", "name": "LANDFIRE_CC_22_mean",
                                               "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 30, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},

                                                ### below here we have non-conus layers (ESI, slope/aspect/srtm)
                                              {"loc": "colorado_esi_2022.tif", "name": "ECOSTRESS_ESI_22",
                                               "base_res": 70, "output_res": 70, "resample_res": 10,
                                               "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                               "override_input_proj": False},
                                              {"loc": "western_conus_aspect.tif", "name": "ASPECT_mean",
                                               "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 30, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                              {"loc": "western_consus_slope.tif", "name": "SLOPE_mean",
                                               "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 30, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                              {"loc": "western_conus_elevation.tif", "name": "SRTM_01_mean",
                                               "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                               "base_res": 30, "output_res": 70, "resample_res": 10,
                                               "override_input_proj": False},
                                          ]}}
    make_config(dbciconf_dict, dbciconf_loc)
    print("done making config")

if sys.argv[1] == "make_ci_config":
    if len(sys.argv) > 2:
        ciconf_name = sys.argv[2]
    else:
        ciconf_name = "default_config"
    ciconf_loc = "../manage_data/configs/ci_" + ciconf_name + ".json"
    print("writing default ci alignment config to", ciconf_loc)
    ### make default CI config
    ciconf_dict = {"core": {"multiprocessing_mode": True,
                            "work_mode": "aws_main",  # {local, aws_main, aws_work, aws_bigbox}
                            "aws_main_mode": True,
                            "aws_work_mode": False,
                            "aws_pems_dir": "~/aws_pems/jega7451.firerx.pem",
                            "aws_workers": ["ubuntu@ec2-54-69-87-217.us-west-2.compute.amazonaws.com",
                                            "ubuntu@ec2-52-37-24-253.us-west-2.compute.amazonaws.com",
                                            "ubuntu@ec2-44-231-201-214.us-west-2.compute.amazonaws.com",
                                            "ubuntu@ec2-34-216-191-56.us-west-2.compute.amazonaws.com"],
                            "aws_primary_box": 0,
                            "aws_inst_directory": "firerx_ml",
                            "aws_env_name": "gdal",
                            "s3_aligned_loc": "s3://firerx.admin/aligned_ci/aligned_wue_california/",
                            "s3_reproj_loc": "s3://firerx.admin/aligned_ci/reprojected_layers/"},

                   "local": {"data_directory_prefix": "../data/",
                             "raster_src": "../data/raw_raster/",
                             "extent_src": "../data/extent/california/",
                             "extent_name": "CA_State",
                             "extent_epsg_override": "3785",
                             "extent_auto_reduce": False, ### TODO --
                             "extent_auto_reduce_by": 1, ### TODO --
                             "trim_step_out": "trimmed/",
                             "reproj_step_out": "reprojected/",
                             "align_step_out": "aligned/",
                             "collection_directory_prefix": "../data/aligned_raster/",
                             "collection_name": "test_align_1/",
                             "raw_data_from_s3": False,
                             "check_local_s3": False,
                             "raw_raster_s3_prefix": "s3://firerx.admin/raw/",
                             "raw_raster_s3_locs": {"PRISM": [("PRISM_tmax_30yr_normal_800mM5_annual_bil.bil", None),
                                                              ("PRISM_tmax_30yr_normal_800mM5_annual_bil.hdr", None),
                                                              ("PRISM_tmax_30yr_normal_800mM5_annual_bil.prj", None),
                                                              ("PRISM_tmax_30yr_normal_800mM5_annual_bil.stx", None),

                                                              ("PRISM_tmean_30yr_normal_800mM5_annual_bil.bil", None),
                                                              ("PRISM_tmean_30yr_normal_800mM5_annual_bil.hdr", None),
                                                              ("PRISM_tmean_30yr_normal_800mM5_annual_bil.prj", None),
                                                              ("PRISM_tmean_30yr_normal_800mM5_annual_bil.stx", None),

                                                              ("PRISM_tmin_30yr_normal_800mM5_annual_bil.bil", None),
                                                              ("PRISM_tmin_30yr_normal_800mM5_annual_bil.hdr", None),
                                                              ("PRISM_tmin_30yr_normal_800mM5_annual_bil.prj", None),
                                                              ("PRISM_tmin_30yr_normal_800mM5_annual_bil.stx", None),

                                                              ("PRISM_vpdmax_30yr_normal_800mM5_annual_bil.bil", None),
                                                              ("PRISM_vpdmax_30yr_normal_800mM5_annual_bil.hdr", None),
                                                              ("PRISM_vpdmax_30yr_normal_800mM5_annual_bil.prj", None),
                                                              ("PRISM_vpdmax_30yr_normal_800mM5_annual_bil.stx", None),

                                                              ("PRISM_vpdmin_30yr_normal_800mM5_annual_bil.bil", None),
                                                              ("PRISM_vpdmin_30yr_normal_800mM5_annual_bil.hdr", None),
                                                              ("PRISM_vpdmin_30yr_normal_800mM5_annual_bil.prj", None),
                                                              ("PRISM_vpdmin_30yr_normal_800mM5_annual_bil.stx", None),

                                                              ("PRISM_ppt_30yr_normal_800mM4_annual_bil.bil", None),
                                                              ("PRISM_ppt_30yr_normal_800mM4_annual_bil.hdr", None),
                                                              ("PRISM_ppt_30yr_normal_800mM4_annual_bil.prj", None),
                                                              ("PRISM_ppt_30yr_normal_800mM4_annual_bil.stx", None)],
                                                    "LANDFIRE": [("LANDFIRE_EVC_01.tif", "landfire_evc_01.tif"),
                                                                 ("LANDFIRE_EVH_01.tif", "landfire_evh_01.tif"),
                                                                 ("LANDFIRE_EVT_01.tif", "landfire_evt_01.tif"),
                                                                 ("LANDFIRE_EVC_14.tif", "landfire_evc_14.tif"),
                                                                 ("LANDFIRE_EVH_14.tif", "landfire_evh_14.tif"),
                                                                 ("LANDFIRE_EVT_14.tif", "landfire_evt_14.tif"),
                                                                 ("LANDFIRE_EVC_16.tif", "landfire_evc_16.tif"),
                                                                 ("LANDFIRE_EVH_16.tif", "landfire_evh_16.tif"),
                                                                 ("LANDFIRE_EVT_16.tif", "landfire_evt_16.tif")],
                                                    "standage": [("conus_age_1km_albers.tif", None)],
                                                    "ECOSTRESS": [("2020_mosaic.tif", "median_WUE_CA.tif")
                                                                  ]},
                             "raw_extent_s3_loc": "extents/california/"},

                   "skip": {"guiding_load": False,
                            "trim": False,
                            "raster_reproj": False,
                            "extent_reproj": False,
                            "resolution_check": False,
                            "raw": False,
                            "rm_existing_TEST": False,
                            "subset_data_TEST": False,
                            "worker_subset_TEST": None,
                            "do_not_align": False,
                            "retrieve_only": False,
                            "skip_checkpoint": 0,
                            "aws_skip_fullsetup": False,  ### TODO???
                            "aws_skip_transfer_raw": True,
                            "aws_skip_transfer": [],
                            "aws_skip_begin": []},

                   "params": {"res_check_tol": 0.001,
                              "output_nodata": -99999,
                              "resampling_mode": "n_meter"},
                   "data": {"align_to_layer": {"loc": "median_WUE_CA.tif",
                                               "name": "ECOSTRESS_WUE",
                                               "resolution": 70,
                                               "input_func": "align_tif",
                                               "override_input_proj": False},
                            "data_info": [{"loc": "conus_age_1km_albers.tif", "name": "TREEAGE_mean",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 1000, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": "5070"},

                                          {"loc": "PRISM_tmax_30yr_normal_800mM5_annual_bil.bil",
                                           "name": "PRISM_TEMPMAX_30_mean",
                                           "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 800, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "PRISM_tmean_30yr_normal_800mM5_annual_bil.bil",
                                           "name": "PRISM_TEMPMEAN_30_mean",
                                           "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 800, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "PRISM_tmin_30yr_normal_800mM5_annual_bil.bil",
                                           "name": "PRISM_TEMPMIN_30_mean",
                                           "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 800, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "PRISM_vpdmax_30yr_normal_800mM5_annual_bil.bil",
                                           "name": "PRISM_VAPORMAX_30_mean",
                                           "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 800, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "PRISM_vpdmin_30yr_normal_800mM5_annual_bil.bil",
                                           "name": "PRISM_VAPORMIN_30_mean",
                                           "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 800, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "PRISM_ppt_30yr_normal_800mM4_annual_bil.bil",
                                           "name": "PRISM_PRECIP_30_mean",
                                           "input_func": "align_bil", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 800, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},

                                          {"loc": "landfire_evc_01.tif", "name": "LANDFIRE_EVC_01_mean",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 30, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "landfire_evh_01.tif", "name": "LANDFIRE_EVH_01_mean",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 30, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "landfire_evt_01.tif", "name": "LANDFIRE_EVT_01_mode",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mode",
                                           "base_res": 30, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "landfire_evc_14.tif", "name": "LANDFIRE_EVC_14_mean",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 30, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "landfire_evh_14.tif", "name": "LANDFIRE_EVH_14_mean",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 30, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "landfire_evt_14.tif", "name": "LANDFIRE_EVT_14_mode",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mode",
                                           "base_res": 30, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "landfire_evc_16.tif", "name": "LANDFIRE_EVC_16_mean",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 30, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "landfire_evh_16.tif", "name": "LANDFIRE_EVH_16_mean",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mean",
                                           "base_res": 30, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          {"loc": "landfire_evt_16.tif", "name": "LANDFIRE_EVT_16_mode",
                                           "input_func": "align_tif", "avg_func": "align_avg", "avg_method": "mode",
                                           "base_res": 30, "output_res": 70, "resample_res": 10,
                                           "override_input_proj": False},
                                          ],
                            "subset": None}}
    make_config(ciconf_dict, ciconf_loc)
    print("done making config")

if sys.argv[1] == "make_mldata_config":
    if len(sys.argv) > 2:
        mldata_name = sys.argv[2]
    else:
        mldata_name = "mldata_default_config"
    mldata_loc = "../manage_data/configs/" + mldata_name + ".json"
    print("writing default mldata alignment config to", mldata_loc)

    mldata_dict = {"core": {"mode": "adjust", ### {pyramid, cube, adjust}
                            "run_name_suffix_default": "_lf22",
                            "run_name_prefix": "../data/ml_sets/",
                            "low_memory_mode": True},
                   "params": {"dimension_override": 1000,
                              "cube_reduce_dim_factor": 1,
                              "load_checkpoint": True,
                              "data_target_proj": "EPSG4326WSG84", ### TODO
                              "split_method": "blocks", ### {blocks, fullrandom}
                              "split_blocks_buffer": 10,
                              "split_blocks_regions": 100,
                              "split_outer_buffer": 2,
                              "partition_n_splits": 1,
                              "partition_test_ratio": 0.3,
                              "partition_val_ratio": 0.2,
                              "exclude": [],#[3, 4, 5, 6, 7, 8, 9, 10, 11],
                              "buffer_fill": -999,
                              "np_random_seed": 201951,
                              "h5_chunk_size": 1000,
                              "parallelize": True,
                              "base_layer": 17},

                   "data": {"data_info": [{"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/SRTM_01_mean_4326_0_0.tif",
                                           "name": "SRTM", "base_res": 30, "xy": "x", "index": 0,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/ASPECT_mean_4326_0_0.tif",
                                           "name": "Aspect", "base_res": 30, "xy": "x", "index": 1,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/SLOPE_mean_4326_0_0.tif",
                                           "name": "Slope", "base_res": 30, "xy": "x", "index": 2,
                                           "type": "numeric"},

                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/LANDFIRE_EVC_22_mean_4326_0_0.tif",
                                           "name": "EVC2022", "base_res": 30, "xy": "x", "index": 3,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/LANDFIRE_EVH_22_mean_4326_0_0.tif",
                                           "name": "EVH2022", "base_res": 30, "xy": "x", "index": 4,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/LANDFIRE_EVT_22_mode_4326_0_0.tif",
                                           "name": "EVT2022", "base_res": 30, "xy": "x", "index": 5,
                                           "type": "categoric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/LANDFIRE_CC_22_mean_4326_0_0.tif",
                                           "name": "CC2022", "base_res": 30, "xy": "x", "index": 6,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/LANDFIRE_FVH_22_mean_4326_0_0.tif",
                                           "name": "FVH2022", "base_res": 30, "xy": "x", "index": 7,
                                           "type": "numeric"},

                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/PRISM_PRECIP_30_mean_4326_0_0.tif",
                                           "name": "Precip", "base_res": 800, "xy": "x", "index": 8,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/PRISM_TEMPMIN_30_mean_4326_0_0.tif",
                                           "name": "TempMIN", "base_res": 800, "xy": "x", "index": 9,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/PRISM_TEMPMEAN_30_mean_4326_0_0.tif",
                                           "name": "TempMEAN", "base_res": 800, "xy": "x", "index": 10,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/PRISM_TEMPMAX_30_mean_4326_0_0.tif",
                                           "name": "TempMAX", "base_res": 800, "xy": "x", "index": 11,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/PRISM_VAPORMIN_30_mean_4326_0_0.tif",
                                           "name": "VaporMIN", "base_res": 800, "xy": "x", "index": 12,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/PRISM_VAPORMAX_30_mean_4326_0_0.tif",
                                           "name": "VaporMAX", "base_res": 800, "xy": "x", "index": 13,
                                           "type": "numeric"},

                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/TREEAGE_mean_4326_0_0.tif",
                                           "name": "StandAge", "base_res": 1000, "xy": "x", "index": 14,
                                           "type": "numeric"},

                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/ECOSTRESS_WUE_22_4326_0_0.tif",
                                           "name": "ECOSTRESSWUE", "base_res": 70, "xy": "y", "index": 15,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/ECOSTRESS_ESI_22_4326_0_0.tif",
                                           "name": "ECOSTRESSESI", "base_res": 70, "xy": "y", "index": 16,
                                           "type": "numeric"},
                                          {"loc": "../data/aligned_raster/colorado_ml_lf22/reprojected/GEDI_AGB_4326_0_0.tif",
                                           "name": "GEDIAGB", "base_res": 1000, "xy": "y", "index": 17,
                                           "type": "numeric"},
                                          ],
                            "test_subset": []}}
    make_config(mldata_dict, mldata_loc)
    print("done making config")

if sys.argv[1] == "make_mlframe_config":
    if len(sys.argv) > 2:
        mlframe_name = sys.argv[2]
    else:
        mlframe_name = "mlframe_default_config"
    mlframe_loc = "../models/configs/" + mlframe_name + ".json"
    print("writing default ci alignment config to", mlframe_loc)

    mlframe_dict = {"core": {"verbosity": 2,
                             "model_dicts_locs": ["../models/configs/f2_baseline.json"],
                             "override_existing_dir": False},
                    }

    make_config(mlframe_dict, mlframe_loc)
    print("done making config")

if sys.argv[1] == "make_model_config":
    if len(sys.argv) > 2:
        model_type = sys.argv[2]
    else:
        print("missing argument for model type.")
        sys.exit(0)

    modelconf_loc = "../models/configs/" + model_type + ".json"
    print("writing default ci alignment config to", modelconf_loc)

    ## cascade_late (1)
    mlframe_dict = {"run_params": {"data_root_dir": "../data/ml_sets/adjust_lf22/",
                                   "make_vis": None,
                                   "train_params":  {"mode": "train", ### {load, loadtrain, train}
                                                     "batch_size": 1000,
                                                     "run_on_folds": [0],
                                                     "n_epochs_default": 150,
                                                     "verbosity": 2,
                                                     "workers": 0,
                                                     "multip": True,
                                                     "callbacks": ["loss", "checkpoint"],
                                                     "save_model": True,
                                                     "compute_metrics": True,
                                                     "metrics_params": ["single"]
                                                    },
                                   },

                    "model_params": {"model_type": "f2_baseline", ### f2 (baseline) c2_early c2_mid c2_late_a c2_late_b
                                     "model_dir": "trained/" + model_type + "_adjust",
                                     "model_name": "f2_baseline_1",
                                     "hyperparams": {"dense_layers": [1600, 1600, 1600],
                                                     "learning_rate": 0.0005,
                                                     "single_task": None,
                                                     "monitor_loss": True,
                                                     "training_loss": "mse",
                                                     "output_block": "cascade"}
                                     },
                    }
    ### multi vit
    """
    mlframe_dict = {"run_params": {"data_root_dir": "../data/ml_sets/pyramid_lf22/",
                                   "make_vis": None,
                                   "train_params": {"mode": "train",  ### {load, loadtrain, train}
                                                    "batch_size": 1000,
                                                    "run_on_folds": [0],
                                                    "n_epochs_default": 1,
                                                    "verbosity": 1,
                                                    "workers": 0,
                                                    "multip": True,
                                                    "callbacks": ["loss", "checkpoint"],
                                                    "save_model": True,
                                                    "compute_metrics": True,
                                                    "metrics_params": ["single"]
                                                    },
                                   },

                    "model_params": {"model_type": "vit",  ### f1 (baseline) c2_early c2_mid c2_late_a c2_late_b vit
                                     "model_dir": "trained/" + model_type + "_pyramid",
                                     "model_name": "vit_pyramid_1",
                                     "hyperparams": {"training_loss": "mse",
                                                     "monitor_loss": True,
                                                     "n_patches": [4, 5, 6],
                                                     "proj_dim": 16,
                                                     "n_heads": 6,
                                                     "t_layers": 6,
                                                     "dense_layers": [1600, 1600, 1600],
                                                     "singletask": None,
                                                     "learning_rate": 0.0005,
                                                     "output_block": "cascade"}
                                     },
                    }"""

    make_config(mlframe_dict, modelconf_loc)
    print("done making config")
