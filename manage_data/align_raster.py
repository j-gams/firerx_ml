### DO IMPORTS
### name == main necessary for multiprocessing?
if __name__ == "__main__":
    print("importing...")
    import sys
    sys.path.append("..")
    import os
    import math
    import numpy as np
    import multiprocessing
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
    from multiprocessing import Pool
    import align_ci_helpers as align
    from utils import utils
    import subprocess
    import time

    ### DEAL WITH PARAMETERS, CONFIG FILES...
    config_dir = "configs/"
    config_prefix = "ci_"
    config_name = "default_config"
    if len(sys.argv) > 2 and sys.argv[2][:6] == "config":
        config_name = sys.argv[2][7:]
    config_loc = config_dir + config_prefix + config_name + ".json"
    ### load config
    config = utils.read_config(config_loc)
    ### distribute config values to parameters...
    print("loaded config file")

    ### CORE PARAMETERS
    ### multiprocess?
    multiprocessing_mode = config["core"]["multiprocessing_mode"]
    ### whether to run things on aws instances (assumes you can login with pem files)
    ### idea is to do trim and reproj locally, then send smaller files to aws for expensive resampling
    ### this one -- for local machine when in aws mode to trim and reproj
    aws_main_mode = config["core"]["aws_main_mode"]
    ### this one -- for remote machines when in aws mode to resample
    aws_work_mode = config["core"]["aws_work_mode"]
    ### aws pem files location, to log in to remote machines
    aws_pems_dir = config["core"]["aws_pems_dir"]
    ### list of elastic ips for aws instances
    aws_workers = config["core"]["aws_workers"]
    n_aws_instances = len(aws_workers)
    ### aws remote instance file location
    aws_inst_directory = config["core"]["aws_inst_directory"]
    ### aws remote instance conda environment name
    aws_env_name= config["core"]["aws_env_name"]
    ### s3 location for aligned layers, reprojected layers... these directories must already exist
    s3_aligned_loc = config["core"]["s3_aligned_loc"]
    s3_reproj_loc = config["core"]["s3_reproj_loc"]

    ### SETUP LOCAL ENVIRONMENT
    ### source raw data
    raster_src = config["local"]["raster_src"]
    extent_src = config["local"]["extent_src"]
    ### extent name
    extent_name = config["local"]["extent_name"]
    extent_src = extent_src + extent_name
    ### if there is an issue with ability to read extent projection...
    extent_epsg_override = config["local"]["extent_epsg_override"]
    ### whether to slice extent into smaller chunks to batch larger samples
    extent_auto_reduce = config["local"]["extent_auto_reduce"]
    ### degree to which extent should be reduced TODO -- tbd
    extent_auto_reduce = config["loc"]["extent_auto_reduce_by"]
    ### step 1 save (trim to CO)
    trim_step_out = config["local"]["trim_step_out"]
    ### step 2 reproject (WGS 84)
    reproj_step_out = config["local"]["reproj_step_out"]
    ### step 3 re-align
    align_step_out = config["local"]["align_step_out"]
    ### put steps under the umbrella of the collection
    collection_name = config["local"]["collection_name"]
    collection_prefix = config["local"]["collection_directory_prefix"] + collection_name
    ### make more useful path variables from parameters
    trim_step_dir = collection_prefix + trim_step_out
    reproj_step_dir = collection_prefix + reproj_step_out
    align_step_dir = collection_prefix + align_step_out
    ### whether to first download data from s3.... useful when running from ec2 machine
    raw_data_from_s3 = config["local"]["raw_data_from_s3"]
    ### where in s3 is the raw raster data?
    raw_raster_s3_prefix = config["local"]["raw_raster_s3_prefix"]
    ### where in s3 is the raw extent data?
    raw_extent_s3_loc = config["local"]["raw_extent_s3_loc"]
    ### where each file is on aws with different directories...
    raw_raster_s3_locs = config["local"]["raw_raster_s3_locs"]
    ### check if data exists on machine before transferring from s3
    check_local_s3 = config["local"]["check_local_s3"]

    ### SKIP EXTRA WORK
    ### whether to skip loading the guiding layer
    skip_guiding_load = config["skip"]["guiding_load"]
    ### whether to skip trimming layers to the extent
    skip_trim = config["skip"]["trim"]
    ### whether to skip raster reprojections
    skip_raster_reproj = config["skip"]["raster_reproj"]
    ### whether to skip reprojecting the extent before trimming
    skip_extent_reproj = config["skip"]["extent_reproj"]
    ### whether to skip the check on reported vs observed resolutions
    skip_resolution_check = config["skip"]["resolution_check"]
    ### skip loading raw x data -- aws worker only
    skip_raw = config["skip"]["raw"]
    ### delete old files on each run -- test only
    rm_existing_TEST = config["skip"]["rm_existing_TEST"]
    ### run only on first file -- test only
    subset_data_TEST = config["skip"]["subset_data_TEST"]
    ### test aws mode on one worker
    worker_subset_TEST = config["skip"]["worker_subset_TEST"]
    if worker_subset_TEST is not None:
        aws_workers = [aws_workers[ii] for ii in worker_subset_TEST]
    ### use to convert all layers to same crs without aligning... prep for ML-ready datasets
    do_not_align = config["skip"]["do_not_align"]
    ### use to gather and retrieve work from aws instances
    retrieve_only = config["skip"]["retrieve_only"]
    ### is this used...?
    skip_checkpoint = config["skip"]["skip_checkpoint"]
    ### use to skip transferring raw data files to aws (speed things up_
    aws_skip_transfer_raw = config["skip"]["aws_skip_transfer_raw"]
    ### use to skip transferring data to aws instances when it already exists there
    aws_skip_transfer = config["skip"]["aws_skip_transfer"]
    ### used for debugging...?
    aws_skip_begin = config["skip"]["aws_skip_begin"]

    ### PARAMETERS
    ### tolerance for resolution reported vs observed check
    res_check_tol = config["params"]["res_check_tol"]
    ### nodata value to use
    output_nodata = config["params"]["output_nodata"]
    ### resampling mode --
    resampling_mode = config["params"]["resampling_mode"]

    ### provide layer info for resampling
    ### note --- base res is only taken for granted for align_to layer
    ### other layers (being aligned) the base res is only for resampling etc.
    align_to_layer = config["data"]["align_to_layer"]
    data_info = config["data"]["data_info"]
    ### test mode option to run on smaller subset of layers
    data_subset_TEST = config["data"]["subset"]
    if data_subset_TEST is not None:
        data_info = [data_info[ii] for ii in data_subset_TEST]

    ### non-test subset...
    subset = [ii for ii in range(len(data_info))]

    print("processing arguments")
    print(sys.argv)
    ### RETRIEVE COMMAND LINE ARGS
    for claid in range(1, len(sys.argv)):
        cla = sys.argv[claid]
        if cla is None:
            continue
        ### CORE PARAMS
        if cla[:8] == "core_mpm":
            multiprocessing_mode = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "core_aws":
            aws_main_mode = (cla[9:] == "True" or cla[9:] == "true")
            print("  - setting subset list to", cla[9:], aws_main_mode)
        if cla[:8] == "core_aw2":
            aws_work_mode = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "core_pem":
            aws_pems_dir = cla[9:]
        if cla[:8] == "core_wkr":
            aws_workers = cla[9:].split(",")
        if cla[:8] == "core_dir":
            aws_inst_directory = cla[9:]
        if cla[:8] == "core_env":
            aws_env_name = cla[9:]
        if cla[:8] == "core_al3":
            s3_aligned_loc = cla[9:]
        if cla[:8] == "core_rj3":
            s3_reproj_loc = cla[9:]
        ### LOCAL PARAMS
        if cla[:8] == "locl_ras":
            raster_src = cla[9:]
            print("  - set raster source to", raster_src)
        if cla[:8] == "locl_exs":
            extent_src = cla[9:]
        if cla[:8] == "locl_tro":
            trim_step_out = cla[9:]
        if cla[:8] == "locl_rjo":
            reproj_step_out = cla[9:]
        if cla[:8] == "locl_cln":
            collection_name = cla[9:]
        if cla[:8] == "locl_clp":
            collection_prefix = cla[9:]
        ### SKIP STEPS
        if cla[:8] == "skip_gld":
            skip_guiding_load = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "skip_trm":
            skip_trim = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "skip_rrj":
            skip_raster_reproj = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "skip_erj":
            skip_extent_reproj = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "skip_res":
            skip_resolution_check = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "skip_raw":
            skip_raw = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "skip_rme":
            rm_existing_TEST = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "skip_cpt":
            skip_checkpoint = int(cla[9:])
        if cla[:8] == "no_align":
            do_not_align = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "retrieve":
            retrieve_only = (cla[9:] == "True" or cla[9:] == "true")
        ### PARAMS
        if cla[:8] == "pram_tol":
            res_check_tol = float(cla[9:])
        if cla[:8] == "pram_ndo":
            output_nodata = float(cla[9:])
        if cla[:8] == "pram_rsm":
            resampling_mode = cla[9:]
        if cla[:8] == "pram_sub":
            subset = [int(subid) for subid in cla[9:].split(",")]
            print("  - setting subset list to", subset)

    ### RUN
    ### diagnostics... summarize what will be run
    print("process summary:")
    print("  - skipping guiding load:", skip_guiding_load)
    print("  - skipping trim:", skip_trim)
    print("  - skipping raster reproj:", skip_raster_reproj)
    print("  - skipping vector reproj:", skip_extent_reproj)
    print("  - skipping layer resolution check:", skip_extent_reproj)
    print("  - skipping layer alignment:", do_not_align)
    print("  - running in retrieve mode:", retrieve_only)
    print("  - running in multiprocessing mode:", multiprocessing_mode)
    print("  - running in core aws mode:", aws_main_mode)
    print("  - running in work aws mode:", aws_work_mode)
    print("  - resampling with ", resampling_mode)
    if rm_existing_TEST:
        print("  - deleting old files (TEST MODE)")
    if subset_data_TEST:
        print("  - only running on first file (TEST MODE)")

    if retrieve_only:
        print("determining allocations...")
        ### compute which files are assigned to which aws instances
        assignment_ro = [ii % n_aws_instances for ii in range(len(data_info))]
        assignment_indiv_ro = [[] for ii in range(n_aws_instances)]
        ### to be changed upon finding/not finding files
        assignment_str_ro = ["" for ii in range(n_aws_instances)]
        results_found_ro = [0 for ii in range(n_aws_instances)]
        results_nofile_ro = [0 for ii in range(n_aws_instances)]
        alldone = True
        for i in range(len(data_info)):
            assignment_indiv_ro[assignment_ro[i]].append(i)
            assignment_str_ro[assignment_ro[i]] += str(i) + ","

        print("checking completion...")
        ### big idea: iterate over aws instances and assigned files, check for their existence
        for i in range(n_aws_instances):
            for j in range(len(assignment_indiv_ro[i])):
                ### need to interact with instances this way to get output
                ### this calls a shell script that returns "exists" if it finds the requested file or "nofile" otherwise
                stepres = subprocess.check_output("sh aws_check.sh " + aws_pems_dir + " " + aws_workers[i] + " " +
                                                  aws_inst_directory + "/data/" + align_step_dir +
                                                  data_info[assignment_indiv_ro[i][j]]["name"] + '.tif ',
                                                  shell=True, text=True)
                if stepres[-7:-1] == "exists":
                    results_found_ro[i] += 1
                elif stepres[-7:-1] == "nofile":
                    results_nofile_ro[i] += 1
            pbar = ""
            for j in range(results_found_ro[i]):
                pbar += "#"
            for j in range(results_nofile_ro[i]):
                pbar += "_"
            for j in range(len(assignment_indiv_ro[i]) - (results_found_ro[i] + results_nofile_ro[i])):
                pbar += " "
            print("  - instance", i, "[" + pbar + "] (" + str(results_found_ro[i]) + "/" + str(len(assignment_indiv_ro[i])) + ")")
            if results_found_ro[i] < len(assignment_indiv_ro[i]):
                alldone = False
        if alldone:
            ### retrieve from aws
            for i in range(n_aws_instances):
                for j in range(len(assignment_indiv_ro[i])):
                    ### copy from aws to local
                    os.system('scp -i "' + aws_pems_dir + '" ' + aws_workers[i] + ':~/' + aws_inst_directory +
                              '/data/' + align_step_dir + data_info[assignment_indiv_ro[i][j]]["name"] + '.tif ' +
                              align_step_dir + data_info[assignment_indiv_ro[i][j]]["name"] + ".tif")
            print("done retrieving files")
            sys.exit(0)
        else:
            print(str(sum(results_found_ro)) + "/" + str(len(data_info)) + " completed")
            print("incomplete, check back later")
            sys.exit(0)

    ### setup environment -- make collection directory
    os.system("mkdir " + collection_prefix)
    ### make trim step working directory
    os.system("mkdir " + trim_step_dir)
    ### make reproj step working directory
    os.system("mkdir " + reproj_step_dir)
    ### make output (aligned) directory
    os.system("mkdir " + align_step_dir)

    ### makes gdal less annoying
    gdal.UseExceptions()

    ### test subset
    if subset_data_TEST:
        data_info = [data_info[1]]
    ### aws subsetting
    subset_list = []
    for i in range(len(subset)):
        subset_list.append(data_info[subset[i]])
    data_info = subset_list

    ### IF WE ARE GETTING RAW DATA FROM S3!!!
    ### if we need to collect raw data from s3...
    if raw_data_from_s3:
        os.system('mkdir ' + raster_src)
        os.system('mkdir ' + extent_src)
        print("transferring extent files from aws s3:")
        os.system('aws s3 cp ' + raw_raster_s3_prefix + raw_extent_s3_loc + extent_name + '.shp ' + extent_src + '.shp')
        os.system('aws s3 cp ' + raw_raster_s3_prefix + raw_extent_s3_loc + extent_name + '.prj ' + extent_src + '.prj')
        os.system('aws s3 cp ' + raw_raster_s3_prefix + raw_extent_s3_loc + extent_name + '.shx ' + extent_src + '.shx')
        print("transferring all listed rasters:")
        for dir_k in raw_raster_s3_locs:
            for i in range(len(raw_raster_s3_locs[dir_k])):
                os.system('aws s3 cp ' + raw_raster_s3_prefix + dir_k + '/' + raw_raster_s3_locs[dir_k][i][0] +
                          ' ' + raster_src + align.munger(raw_raster_s3_locs[dir_k][i]))

        ### ok... good to go?

    ### load unaligned raw data
    xraster = []
    xr_proj = []
    xr_ndvals = []
    xr_rastersize = []
    xr_crs = []
    xr_nparray = []
    if aws_work_mode:
        align_to_layer["loc"] = align_to_layer["name"] + ".tif"

    ### LOAD RAW GUIDING LAYER DATA
    if not skip_guiding_load:
        print("loading guiding layer: " + align_to_layer["loc"])
        yraster = gdal.Open(raster_src + align_to_layer["loc"])
        yrband = yraster.GetRasterBand(1)
        yr_ndval = yrband.GetNoDataValue()
        yr_size = (yraster.RasterXSize, yraster.RasterYSize)
        yulh, yph, _, yulv, _, ypv = yraster.GetGeoTransform()
        ypv = abs(ypv)
        yr_proj = yraster.GetProjection()
        yr_crs = (yulh, yulv, yph, ypv)
        yr_nparray = yraster.ReadAsArray().transpose()
        yr_full_geotrans = yraster.GetGeoTransform()

        yr_epsg = osr.SpatialReference(wkt=yr_proj).GetAttrValue('AUTHORITY', 1)

        ### expect 4326 for WUE
        print("  - guiding layer epsg", yr_epsg)

    ### LOAD EXTENT ... but only if not skip_trim
    if not skip_trim:
        print("loading extent")
        extent = gdal.OpenEx(extent_src + ".shp", sibling_files=[extent_src + ".prj"])
        extent_proj = extent.GetLayer().GetSpatialRef()
        extent_epsg = extent_proj.GetAttrValue('AUTHORITY', 1)
        if extent_epsg_override is not False:
            print("overriding extent epsg from", extent_epsg, "to", extent_epsg_override)
            extent_epsg = extent_epsg_override
        print("extent epsg", extent_epsg)
        ### reproject to match baseline proj
        if not skip_extent_reproj:
            print("reprojecting and saving extent shapefile in original location")
            if rm_existing_TEST:
                os.system('rm ' + extent_src + '_' + str(yr_epsg) + '.shp')
            os.system(
                'ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:' + str(yr_epsg) + ' -s_srs EPSG:' + str(extent_epsg) + ' ' +
                extent_src + '_' + str(yr_epsg) + '.shp ' + extent_src + '.shp')
            correct_proj_extent = extent_src + "_" + str(yr_epsg) + ".shp"
            new_proj_base = extent_src
        else:
            correct_proj_extent = extent_src + ".shp"
            ### this used to be a direct reference to file -- why? changed to src
            new_proj_base = extent_src
        if not skip_guiding_load:
            print("trimming guiding file")
            os.system('gdalwarp -cutline ' + new_proj_base + '_' + yr_epsg + '.shp -crop_to_cutline -dstnodata "' +
                      str(yr_ndval) + '" ' + raster_src + align_to_layer["loc"] + ' ' + trim_step_dir +
                      align_to_layer["name"] + ".tif")

        print("reloading guiding layer: " + trim_step_dir + align_to_layer["name"] + ".tif")
        yraster = gdal.Open(trim_step_dir + align_to_layer["name"] + ".tif")
        yrband = yraster.GetRasterBand(1)
        yr_ndval = yrband.GetNoDataValue()
        yr_size = (yraster.RasterXSize, yraster.RasterYSize)
        yulh, yph, _, yulv, _, ypv = yraster.GetGeoTransform()
        ypv = abs(ypv)
        yr_proj = yraster.GetProjection()
        yr_crs = (yulh, yulv, yph, ypv)
        yr_nparray = yraster.ReadAsArray().transpose()
        yr_full_geotrans = yraster.GetGeoTransform()

        yr_epsg = osr.SpatialReference(wkt=yr_proj).GetAttrValue('AUTHORITY', 1)

        ### expect 4326 for WUE
        print("  - guiding layer epsg doublecheck", yr_epsg)
    else:
        print("skipping trimming process")

    ### LOAD RAW X DATA
    print("loading raw raster data")
    for i in range(len(data_info)):
        if not skip_raw:
            print("  - loading raw " + data_info[i]["loc"] + " data (" + str(i) + ")...")
            temp_raster = gdal.Open(raster_src + data_info[i]["loc"])
            temp_band = temp_raster.GetRasterBand(1)
            temp_proj = temp_raster.GetProjection()
            temp_ndval = temp_band.GetNoDataValue()
            _, temp_hres, _, _, _, temp_vres = temp_raster.GetGeoTransform()
        else:
            print("  - skipping raw data load")

        ### IF TRIM --- TRIM TO CO AOI
        if not skip_trim:
            ### determine projection for shapefile...
            ### below 2 lines -- get epsg number from this raster layer
            wktproj = osr.SpatialReference(wkt=temp_proj)
            layer_epsg = wktproj.GetAttrValue('AUTHORITY', 1)
            print("  - layer projection is EPSG:" + str(layer_epsg))
            ### deal with layers with messed up metadata
            if data_info[i]["override_input_proj"] is not False:
                print("  - overriding input layer projection to EPSG:" + str(data_info[i]["override_input_proj"]))
                layer_epsg = str(data_info[i]["override_input_proj"])
            if rm_existing_TEST:
                os.system('rm ' + new_proj_base + '_' + layer_epsg + '.shp')
                os.system('rm ' + trim_step_out + data_info[i]["name"] + '.tif')
            ### reproject extent to current layer raw projection
            os.system('ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:' + str(layer_epsg) + ' -s_srs EPSG:' +
                      str(extent_epsg) + ' ' + new_proj_base + '_' + str(layer_epsg) + '.shp ' + extent_src + '.shp')
            print("  - reprojected shapefile to x proj: EPSG", layer_epsg)
            ### trim to extent in raw projection
            os.system('gdalwarp -cutline ' + new_proj_base + '_' + layer_epsg + '.shp -crop_to_cutline -dstnodata "' +
                      str(temp_ndval) + '" ' + raster_src + data_info[i]["loc"] + ' ' + trim_step_dir +
                      data_info[i]["name"] + ".tif")
            print("  - saved trimmed file")

        ### IF PROJECTION IS NOT BASE PROJECTION --- NEED TO REPROJECT!
        if not skip_raster_reproj:
            if layer_epsg == yr_epsg:
                print("  - no reprojection required -- x epsg = ", layer_epsg, "-- guiding epsg = ", yr_epsg)
                ### check whether reported resolution is consistent with actual pixel widths
                if not skip_resolution_check:
                    print("  - verifying reported layer resolution")
                    ### reported pixel widths -- base * fraction of base reported to layer reported
                    reported_xres = yph * (data_info[i]["base_res"] / align_to_layer["resolution"])
                    reported_yres = ypv * (data_info[i]["base_res"] / align_to_layer["resolution"])
                    actual_xres = temp_hres
                    actual_yres = temp_vres
                    if abs(reported_xres - actual_xres) > res_check_tol or abs(
                            reported_yres - actual_yres) > res_check_tol:
                        ### fails tolerance check
                        print("  - layer resolution outside of tolerance -- reported", reported_xres, reported_yres,
                              "-- actual --", actual_xres, actual_yres)
                        ### need to resample to reported resolutions
                        gdal.Warp(reproj_step_dir + data_info[i]["name"] + '.tif',
                                  trim_step_dir + data_info[i]["name"] + '.tif', xRes=reported_xres,
                                  yRes=reported_yres)
                    else:
                        ### passes tolerance check -- no action required
                        print("  - layer resolution within tolerance -- reported", reported_xres, reported_yres,
                              "-- actual --", actual_xres, actual_yres)
                        os.system(
                            'mv ' + trim_step_dir + data_info[i]["name"] + '.tif ' + reproj_step_dir + data_info[i][
                                "name"] + '.tif')
                else:
                    ### move file along to next checkpoint
                    os.system(
                        'mv ' + trim_step_dir + data_info[i]["name"] + '.tif ' + reproj_step_dir + data_info[i][
                            "name"] + '.tif')
            else:
                print("  - reprojection required -- x epsg = ", layer_epsg, "-- guiding epsg = ", yr_epsg)
                if rm_existing_TEST:
                    os.system('rm ' + reproj_step_dir + data_info[i]["name"] + '.tif')
                ### compute desired pixel resolutions in consistent crs by computing ratio
                xres_frac = yph * (data_info[i]["base_res"] / align_to_layer["resolution"])
                yres_frac = ypv * (data_info[i]["base_res"] / align_to_layer["resolution"])
                ### reproject to y projection and resample at appropriate resolution
                gdal.Warp(reproj_step_dir + data_info[i]["name"] + '.tif',
                          trim_step_dir + data_info[i]["name"] + '.tif', xRes=xres_frac, yRes=yres_frac,
                          dstSRS='EPSG:' + yr_epsg)
                print("  - saved reprojected layer")

        if not aws_main_mode:
            ### Trimming and reprojection are complete
            ### load trimmed and reprojected rasters
            print("  - loading fixed file...")
            ### open file with gdal
            xraster.append(gdal.Open(reproj_step_dir + data_info[i]["name"] + '.tif'))
            ### get data from first raster band
            tdata_rband = xraster[-1].GetRasterBand(1)
            ### record projection
            xr_proj.append(xraster[-1].GetProjection())
            ### get nodata value from raster band
            xr_ndvals.append(tdata_rband.GetNoDataValue())
            ### get raster grid size
            xr_rastersize.append((xraster[-1].RasterXSize, xraster[-1].RasterYSize))
            ### get crs parameters from raster
            ulh, ph, _, ulv, _, pv = xraster[-1].GetGeoTransform()
            pv = abs(pv)
            ### record simple crs
            xr_crs.append((ulh, ulv, ph, pv))
            ### extract np array from raster
            xr_nparray.append(xraster[-1].ReadAsArray().transpose())
            print("  - loaded fixed file")

    if do_not_align:
        print("quitting after completing basic projection work")
        sys.exit(0)


    ### time to deal with actual alignment
    if aws_main_mode:
        ### at this point... done with setup work. If running in aws mode, need to
        ### - determine which instances should do which files
        ### - move files to aws instances
        ### - setup environment on aws instances
        ### - run this on aws and terminate
        ### - do work
        ### - move files to s3

        print("preparing to move work to aws...")

        ### assign files to aws instances
        assignment = [ii % n_aws_instances for ii in range(len(data_info))]
        assignment_indiv = [[] for ii in range(n_aws_instances)]
        assignment_str = ["" for ii in range(n_aws_instances)]

        for i in range(len(data_info)):
            assignment_indiv[assignment[i]].append(i)
            assignment_str[assignment[i]] += str(i) + ","

        ### perform work common to each instance (directory structure, files, etc.)
        for i in range(n_aws_instances):
            ### set up environment on aws instance
            print("  - setting up aws instance", i)

            os.system('ssh -i "' + aws_pems_dir + '" ' + aws_workers[i] +
                      ' "cd ~; mkdir ' + aws_inst_directory + '; cd ' + aws_inst_directory +
                      '; mkdir data; mkdir manage_data; mkdir manage_data/configs; mkdir utils; mkdir data/aligned_raster; mkdir data/' +
                      collection_prefix + '; mkdir data/' + trim_step_dir + '; mkdir data/' + reproj_step_dir +
                      '; mkdir data/' + align_step_dir + '; mkdir data/' + raster_src + '"')

            ### move shared files to each instance
            ### y geotif
            if i not in aws_skip_transfer:
                print("  - transferring files to aws instance", i)
                ### transfer raw align_to file... for the purposes of moving to s3?
                os.system('scp -i "' + aws_pems_dir + '" ' + trim_step_dir + align_to_layer["name"] + ".tif" + ' ' +
                          aws_workers[i] + ':~/' + aws_inst_directory + '/data/' + raster_src)
                ### this file (align_raster.py)
                os.system('scp -i "' + aws_pems_dir + '" ./align_raster.py ' +
                          aws_workers[i] + ':~/' + aws_inst_directory + '/manage_data/')
                ### helper file (align_ci_helpers.py)
                os.system('scp -i "' + aws_pems_dir + '" ./align_ci_helpers.py ' +
                          aws_workers[i] + ':~/' + aws_inst_directory + '/manage_data/')
                ### config file ...
                os.system('scp -i "' + aws_pems_dir + '" ' + config_loc + ' ' +
                          aws_workers[i] + ':~/' + aws_inst_directory + '/manage_data/' + config_loc)
                ### utils file ...
                os.system('scp -i "' + aws_pems_dir + '" ../utils/utils.py ' +
                          aws_workers[i] + ':~/' + aws_inst_directory + '/utils/')

                ### move assigned files to instance
                for j in range(len(assignment_indiv[i])):
                    os.system(
                        'scp -i "' + aws_pems_dir + '" ' + reproj_step_dir + data_info[assignment_indiv[i][j]]["name"]
                        + '.tif ' + aws_workers[i] + ':~/' + aws_inst_directory + '/data/' + reproj_step_dir)
                    if not aws_skip_transfer_raw:
                        os.system(
                            'scp -i "' + aws_pems_dir + '" ' + raster_src + data_info[assignment_indiv[i][j]][
                                "loc"]
                            + ' ' + aws_workers[i] + ':~/' + aws_inst_directory + '/data/' + raster_src)
            else:
                print("  - skipping transfer to", i)

            ### run files with special parameters
            ### - core aws false, worker aws true
            ### - new raster locations, trim outputs, reprojection outputs
            ### - skip trim, reprojection, extent reproj., resolution check, raw load
            ### - subset
            ### sh aws_psychic.sh ~/aws_pems/jega7451.firerx.pem ubuntu@ec2-54-69-87-217.us-west-2.compute.amazonaws.com firerx_ml/manage_data/ 0,4,8,12 0
            print("    - starting work on aws instance", i)
            if i not in aws_skip_begin:
                print("  - opening subprocess on", i)
                subprocess.Popen(["sh", "aws_psychic.sh", str(aws_pems_dir), str(aws_workers[i]),
                                  str(aws_inst_directory) + "/manage_data/", str(assignment_str[i][:-1]), str(i)])
            else:
                print("    - skip aws start")

        print("closing local work. resampling continuing on aws instances.")
        print("waiting a few seconds to be safe...")
        time.sleep(15)
        sys.exit(0)

    if multiprocessing_mode:
        print("available cpus", multiprocessing.cpu_count())
        print("first raster shape", xr_nparray[0].shape)

    driver = gdal.GetDriverByName("GTiff")

    ### deal with multiprocessing
    ### iterate over layers to align
    print("processing layers...")
    for i in range(len(data_info)):
        ### basic resampling parameters
        guiding_layer_res = align_to_layer["resolution"]
        layer_original_res = data_info[i]["base_res"]
        layer_output_res = data_info[i]["output_res"]
        layer_sampling_res = data_info[i]["resample_res"]
        layer_original_nodata = xr_ndvals[i]
        layer_realign_nodata = output_nodata
        layer_avg_func = data_info[i]["avg_func"]

        ### compute total size of grid in meters
        total_x_meters = align_to_layer["resolution"] * yr_size[0]
        total_y_meters = align_to_layer["resolution"] * yr_size[1]

        ### compute size of output from desired resolution and total meters
        output_x_size = math.ceil(total_x_meters / data_info[i]["output_res"])
        output_y_size = math.ceil(total_x_meters / data_info[i]["output_res"])

        ### compute sampling size... target res // resample_res
        sample_grid_size = math.ceil(data_info[i]["output_res"] / data_info[i]["resample_res"])

        ### target crs -- same ulh, ulv as y crs, widths defined by desired dim
        target_crs = [yulh, yulv, (data_info[i]["output_res"]/align_to_layer["resolution"] * yph), (data_info[i]["output_res"]/align_to_layer["resolution"] * ypv)]

        #slice width, sample_grid_size, layer_crs, target_crs, layer_nodata_val, oob_nodata_val, avg_method
        resample_params = [output_y_size, sample_grid_size, xr_crs[i], target_crs, xr_ndvals[i], output_nodata, data_info[i]["avg_method"], resampling_mode]
        map_params = []
        print(output_x_size)
        if multiprocessing_mode:
            ### determine number of cpus available for multiprocessing purposes
            ncpu = multiprocessing.cpu_count()
        else:
            ncpu = 1
        for j in range(ncpu):
            map_params.append([align.chunk_list(output_x_size, ncpu, j), xr_nparray[i], resample_params])

        ### generate feed parameters
        ### idea is to slice layer into ncpu sections
        ### each iterates over a region to resample
        ### returns subset which is then fixed together

        if multiprocessing_mode:
            ### multiprocess each slice of larger raster
            with Pool(None) as mpool:
                result_arr = mpool.map(align.slice_batch, map_params)
        else:
            result_arr = align.slice_batch(map_params)

        reconstructed_result = []
        for arr_i in result_arr:
            reconstructed_result.extend(arr_i)
            del arr_i

        ### create geotif
        layer_geotif = np.array(reconstructed_result)
        print("  - layer geotif shape:", layer_geotif.shape)

        ### full useful crs
        layer_full_crs = (target_crs[0], target_crs[2], 0, target_crs[1], 0, -target_crs[3])

        ### obtain layer geotif
        driver = gdal.GetDriverByName("GTiff")
        outname = collection_prefix + "/" + data_info[i]["name"] + ".tif"
        layer_out = driver.Create(outname, layer_geotif.shape[0], layer_geotif.shape[1], 1,
                                  gdal.GDT_Float32)
        layer_out.SetGeoTransform(layer_full_crs)
        layer_out.SetProjection(yr_proj)

        ### save layer geotif
        layer_out.GetRasterBand(1).WriteArray(layer_geotif.transpose())
        layer_out.GetRasterBand(1).SetNoDataValue(output_nodata)
        layer_out.FlushCache()

        print("  - saved layer", data_info[i]["name"])

        ### now, if in aws, send files to s3
        if aws_work_mode:
            ### move reprojected layers
            print("  - moving reprojected layer to aws s3")
            os.system('aws s3 cp ' + reproj_step_out + data_info[i]["name"] + '.tif ' + s3_reproj_loc + data_info[i]["name"] + '.tif')
            ### move aligned layer
            print("  - moving realigned layer to aws s3")
            os.system('aws s3 cp ' + outname + ' ' + s3_aligned_loc + data_info[i]["name"] + '.tif')
    print("work complete")