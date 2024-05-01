### DO IMPORTS
### name == main necessary for multiprocessing?
if __name__ == "__main__":
    print("importing...")
    import sys
    import os
    import math
    import numpy as np
    import multiprocessing
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
    from multiprocessing import Pool
    import align_ci_helpers as align
    import subprocess
    import time

    # box1 verified ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-44-237-87-156.us-west-2.compute.amazonaws.com
    # box2 verified ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-54-69-87-217.us-west-2.compute.amazonaws.com
    # box3 verified ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-52-37-24-253.us-west-2.compute.amazonaws.com
    # box4 verified ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-44-231-201-214.us-west-2.compute.amazonaws.com
    # box5 verified ssh -i "aws_pems/jega7451.firerx.pem" ubuntu@ec2-34-216-191-56.us-west-2.compute.amazonaws.com

    ### CORE PARAMETERS
    ### multiprocess?
    multiprocessing_mode = True
    ### whether to run things on aws instances (assumes you can login with pem files)
    ### idea is to do trim and reproj locally, then send smaller files to aws for expensive resampling
    ### this one -- for main
    aws_main_mode = False#True
    ### this one -- for workers
    aws_work_mode = False
    ### aws pem files location
    aws_pems_dir = "~/aws_pems/jega7451.firerx.pem"
    ### list of elastic ips for aws instances
    aws_workers = ["ubuntu@ec2-44-237-87-156.us-west-2.compute.amazonaws.com",
                   "ubuntu@ec2-54-69-87-217.us-west-2.compute.amazonaws.com",
                   "ubuntu@ec2-52-37-24-253.us-west-2.compute.amazonaws.com",
                   "ubuntu@ec2-44-231-201-214.us-west-2.compute.amazonaws.com",
                   "ubuntu@ec2-34-216-191-56.us-west-2.compute.amazonaws.com"]
    n_aws_instances = len(aws_workers)
    ### instance file location
    aws_inst_directory = "align_work"
    ### instance conda environment name
    aws_env_name = "gdal"
    ### s3 location for aligned layers, reprojected layers
    s3_aligned_loc = "s3://firerx.admin/aligned_data/realigned_wue_colorado/"
    s3_reproj_loc = "s3://firerx.admin/aligned_data/reprojected_layers/"

    ### SETUP LOCAL ENVIRONMENT
    ### source raw data
    raster_src = "../data/"
    extent_src = "../data/awsdata/aoi_co/Colorado_State_Boundary"
    ### step 1 save (trim to CO)
    trim_step_out = "../data/raster_2/temp_trim/"
    ### step 2 reproject (WGS 84)
    reproj_step_out = "../data/raster_2/temp_reproj/"
    ### step 3 re-align
    collection_name = "pyramid_raster_2"
    collection_prefix = "../data/" + collection_name

    ### SKIP EXTRA WORK
    skip_trim = False#True
    skip_raster_reproj = False#True
    skip_extent_reproj = False#True
    skip_resolution_check = False#True
    skip_raw = False # skip loading raw x data -- aws worker only
    rm_existing_TEST = False  # delete old files on each run -- test only
    subset_data_TEST = False  # run only on first file -- test only
    do_not_align = True # use to convert all layers to same crs... not align
    skip_checkpoint = 0
    aws_skip_transfer = [0, 1, 2, 3, 4]
    aws_skip_begin = [0, 1]

    ### PARAMETERS
    res_check_tol = 0.001
    output_nodata = -99999
    resampling_mode = "n_meter"

    ### provide layer info for resampling
    ### note --- base res is only taken for granted for align_to layer
    ### other layers (being aligned) the base res is only for resampling etc.
    align_to_layer = {"loc": "raster/gedi_agforestbiomass_clipped_co.tif", "name": "GEDI_AGB", "resolution": 1000,
                      "input_func": align.tif}

    data_info = [{"loc": "geotifs_raw/PRISM_tmax_30yr_normal_800mM5_annual_bil.bil", "name": "PRISM_TEMPMAX_30_mean",
                  "input_func": align.bil, "avg_func": align.avg,
                  "base_res": 800, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/PRISM_tmean_30yr_normal_800mM5_annual_bil.bil", "name": "PRISM_TEMPMEAN_30_mean",
                  "input_func": align.bil, "avg_func": align.avg,
                  "base_res": 800, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/PRISM_tmin_30yr_normal_800mM5_annual_bil.bil", "name": "PRISM_TEMPMIN_30_mean",
                  "input_func": align.bil, "avg_func": align.avg,
                  "base_res": 800, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/PRISM_vpdmax_30yr_normal_800mM5_annual_bil.bil",
                  "name": "PRISM_VAPORMAX_30_mean", "input_func": align.bil, "avg_func": align.avg,
                  "base_res": 800, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/PRISM_vpdmax_30yr_normal_800mM5_annual_bil.bil",
                  "name": "PRISM_VAPORMIN_30_mean", "input_func": align.bil, "avg_func": align.avg,
                  "base_res": 800, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/PRISM_ppt_30yr_normal_800mM4_annual_bil.bil", "name": "PRISM_PRECIP_30_mean",
                  "input_func": align.bil, "avg_func": align.avg,
                  "base_res": 800, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evc_01.tif", "name": "LANDFIRE_EVC_01_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evh_01.tif", "name": "LANDFIRE_EVH_01_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evt_01.tif", "name": "LANDFIRE_EVT_01_mode",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mode", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evc_12.tif", "name": "LANDFIRE_EVC_12_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evh_12.tif", "name": "LANDFIRE_EVH_12_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evc_14.tif", "name": "LANDFIRE_EVC_14_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evh_14.tif", "name": "LANDFIRE_EVH_14_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evt_14.tif", "name": "LANDFIRE_EVT_14_mode",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mode", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evc_16.tif", "name": "LANDFIRE_EVC_16_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evh_16.tif", "name": "LANDFIRE_EVH_16_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/landfire_evt_16.tif", "name": "LANDFIRE_EVT_16_mode",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mode", "resample_res": 10},
                 {"loc": "geotifs_raw/Aspect_Colorado.tif", "name": "ASPECT_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/merged_SRTM.tif", "name": "SRTM_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 {"loc": "geotifs_raw/slope_colo_fixed.tif", "name": "SLOPE_mean",
                  "input_func": align.tif, "avg_func": align.avg,
                  "base_res": 30, "output_res": 70, "avg_method": "mean", "resample_res": 10},
                 ]
    #data_info = data_info[-1:]
    #aws_workers = [aws_workers[1]]
    subset = [ii for ii in range(len(data_info))]
    print(sys.argv)
    ### RETRIEVE COMMAND LINE ARGS
    for claid in range(1, len(sys.argv)):
        cla = sys.argv[claid]
        ### CORE PARAMS
        if cla[:8] == "core_mpm":
            multiprocessing_mode = (cla[9:] == "True" or cla[9:] == "true")
        if cla[:8] == "core_aws":
            aws_main_mode = (cla[9:] == "True" or cla[9:] == "true")
            print("setting subset list to", cla[9:], aws_main_mode)
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
            print("set raster source to", raster_src)
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
        ### PARAMS
        if cla[:8] == "pram_tol":
            res_check_tol = float(cla[9:])
        if cla[:8] == "pram_ndo":
            output_nodata = float(cla[9:])
        if cla[:8] == "pram_rsm":
            resampling_mode = cla[9:]
        if cla[:8] == "pram_sub":
            subset = [int(subid) for subid in cla[9:].split(",")]
            print("setting subset list to", subset)

    ### AWS TESTING STUFF
    """if aws_work_mode:
        print("testing")
        for i in range(10000):
            skip_checkpoint += i
        print("terminating box work.... success??")
        sys.exit(0)"""

    ### RUN
    ### diagnostics
    print("skipping trim:", skip_trim)
    print("skipping raster reproj:", skip_raster_reproj)
    print("skipping vector reproj:", skip_extent_reproj)
    print("skipping layer resolution check:", skip_extent_reproj)
    print("running in multiprocessing mode:", multiprocessing_mode)
    print("running in core aws mode:", aws_main_mode)
    print("running in work aws mode:", aws_work_mode)
    print("resampling with ", resampling_mode)
    if rm_existing_TEST:
        print("deleting old files (TEST MODE)")
    if subset_data_TEST:
        print("only running on first file (TEST MODE)")

    os.system("mkdir " + collection_prefix)

    gdal.UseExceptions()

    ### test subset
    if subset_data_TEST:
        data_info = [data_info[1]]
    ### aws subsetting
    #subset = [ii for ii in range(len(data_info))]
    subset_list = []
    for i in range(len(subset)):
        subset_list.append(data_info[subset[i]])
    data_info = subset_list

    print("checking data info", data_info)

    ### load unaligned raw data
    xraster = []
    xr_proj = []
    xr_ndvals = []
    xr_rastersize = []
    xr_crs = []
    xr_nparray = []

    ### LOAD RAW Y DATA
    print("loading " + align_to_layer["loc"] + " data...")
    yraster = align_to_layer["input_func"](raster_src + align_to_layer["loc"])  # gdal.Open(raster_src + align_to_layer["loc"])
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

    print("y epsg", yr_epsg)

    ### LOAD EXTENT ... but only if not skip_trim
    if not skip_trim:
        print("loading extent")
        extent = gdal.OpenEx(extent_src + ".shp", sibling_files=[extent_src + ".prj"])
        extent_proj = extent.GetLayer().GetSpatialRef()
        # print("extent proj", extent_proj)
        # print(type(extent_proj))
        ### reproject to match baseline proj
        if not skip_extent_reproj:
            print("reprojecting and saving extent shapefile in original location")
            if rm_existing_TEST:
                os.system('rm ' + extent_src + '_4326.shp')
            os.system(
                'ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:4326 -s_srs EPSG:4269 ' + extent_src + '_4326.shp ' + extent_src + '.shp')
            correct_proj_extent = extent_src + "_4326.shp"
            new_proj_base = extent_src
        else:
            correct_proj_extent = extent_src + ".shp"
            new_proj_base = "../data/awsdata/aoi_co/Colorado_State_Boundary"
    else:
        print("skipping trimming process")

    ### LOAD RAW X DATA
    print("loading raw raster data")
    for i in range(len(data_info)):
        if not skip_raw:
            print("- loading raw " + data_info[i]["loc"] + " data (" + str(i) + ")...")
            temp_raster = gdal.Open(raster_src + data_info[i]["loc"])
            temp_band = temp_raster.GetRasterBand(1)
            temp_proj = temp_raster.GetProjection()
            temp_ndval = temp_band.GetNoDataValue()
            _, temp_hres, _, _, _, temp_vres = temp_raster.GetGeoTransform()
        else:
            print("skipping raw data load")

        ### IF TRIM --- TRIM TO CO AOI
        if not skip_trim:
            ### determine projection for shapefile...
            # print(xr_proj[-1])
            # print(type(xr_proj[-1]))
            ### below 2 lines -- get epsg number from this raster layer
            wktproj = osr.SpatialReference(wkt=temp_proj)
            epsg = wktproj.GetAttrValue('AUTHORITY', 1)
            # print(epsg)
            if rm_existing_TEST:
                os.system('rm ' + new_proj_base + '_' + epsg + '.shp')
                os.system('rm ' + trim_step_out + data_info[i]["name"] + '.tif')
            ### reproject extent to current layer raw projection
            os.system(
                'ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:' + epsg + ' -s_srs EPSG:4326 ' + new_proj_base + '_' + epsg + '.shp ' + correct_proj_extent)
            print("reprojected shapefile to x proj: EPSG", epsg)
            ### trim to extent in raw projection
            os.system(
                'gdalwarp -cutline ' + new_proj_base + '_' + epsg + '.shp -crop_to_cutline -dstnodata "' + str(
                    temp_ndval) + '" ' + raster_src + data_info[i]["loc"] + ' ' + trim_step_out + data_info[i][
                    "name"] + ".tif")
            print("saved trimmed file")

        ### IF PROJECTION IS NOT BASE PROJECTION --- NEED TO REPROJECT!
        if not skip_raster_reproj:
            if epsg == yr_epsg:
                print("no reprojection required -- x epsg = ", epsg, "-- y epsg = ", yr_epsg)
                ### check whether reported resolution is consistent with actual pixel widths
                if not skip_resolution_check:
                    print("verifying reported layer resolution")
                    ### reported pixel widths -- base * fraction of base reported to layer reported
                    reported_xres = yph * (data_info[i]["base_res"] / align_to_layer["resolution"])
                    reported_yres = ypv * (data_info[i]["base_res"] / align_to_layer["resolution"])
                    actual_xres = temp_hres
                    actual_yres = temp_vres
                    if abs(reported_xres - actual_xres) > res_check_tol or abs(
                            reported_yres - actual_yres) > res_check_tol:
                        ### fails tolerance check
                        print("layer resolution outside of tolerance -- reported", reported_xres, reported_yres,
                              "-- actual --", actual_xres, actual_yres)
                        ### need to resample to reported resolutions
                        gdal.Warp(reproj_step_out + data_info[i]["name"] + '.tif',
                                  trim_step_out + data_info[i]["name"] + '.tif', xRes=reported_xres,
                                  yRes=reported_yres)
                    else:
                        ### passes tolerance check -- no action required
                        print("layer resolution within tolerance -- reported", reported_xres, reported_yres,
                              "-- actual --", actual_xres, actual_yres)
                        os.system(
                            'mv ' + trim_step_out + data_info[i]["name"] + '.tif ' + reproj_step_out + data_info[i][
                                "name"] + '.tif')
                else:
                    ### move file along to next checkpoint
                    os.system(
                        'mv ' + trim_step_out + data_info[i]["name"] + '.tif ' + reproj_step_out + data_info[i][
                            "name"] + '.tif')
            else:
                print("reprojection required -- x epsg = ", epsg, "-- y epsg = ", yr_epsg)
                if rm_existing_TEST:
                    os.system('rm ' + reproj_step_out + data_info[i]["name"] + '.tif')
                ### compute desired pixel resolutions in consistent crs by computing ratio
                xres_frac = yph * (data_info[i]["base_res"] / align_to_layer["resolution"])
                yres_frac = ypv * (data_info[i]["base_res"] / align_to_layer["resolution"])
                ### reproject to y projection and resample at appropriate resolution
                gdal.Warp(reproj_step_out + data_info[i]["name"] + '.tif',
                          trim_step_out + data_info[i]["name"] + '.tif', xRes=xres_frac, yRes=yres_frac,
                          dstSRS='EPSG:' + yr_epsg)
                print("saved reprojected layer")

        if not aws_main_mode:
            ### Trimming and reprojection are complete
            ### load trimmed and reprojected rasters
            print("loading fixed file...")
            ### open file with gdal
            xraster.append(gdal.Open(reproj_step_out + data_info[i]["name"] + '.tif'))
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
            print("loaded fixed file")

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
            print("setting up aws instance", i)
            os.system('ssh -i "' + aws_pems_dir + '" ' + aws_workers[i] +
                      ' "cd ~; mkdir ' + aws_inst_directory + '; cd ' + aws_inst_directory +
                      '; mkdir raster; mkdir geotifs_raw"')
            ### move shared files to each instance
            ### y geotif
            if i not in aws_skip_transfer:
                print("transferring files to aws instance", i)
                os.system('scp -i "' + aws_pems_dir + '" ' + raster_src + align_to_layer["loc"] + ' ' +
                          aws_workers[i] + ':~/' + aws_inst_directory + '/raster/')
                ### this file (align_raster.py)
                os.system('scp -i "' + aws_pems_dir + '" ./align_raster.py ' +
                          aws_workers[i] + ':~/' + aws_inst_directory + '/')
                ### helper file (align_ci_helpers.py)
                os.system('scp -i "' + aws_pems_dir + '" ./align_ci_helpers.py ' +
                          aws_workers[i] + ':~/' + aws_inst_directory + '/')

                ### move assigned files to instance
                for j in range(len(assignment_indiv[i])):
                    os.system(
                        'scp -i "' + aws_pems_dir + '" ' + reproj_step_out + data_info[assignment_indiv[i][j]]["name"]
                        + '.tif ' + aws_workers[i] + ':~/' + aws_inst_directory + '/geotifs_raw/')
            else:
                print("skipping transfer to", i)

            ### run files with special parameters
            ### - core aws false, worker aws true
            ### - new raster locations, trim outputs, reprojection outputs
            ### - skip trim, reprojection, extent reproj., resolution check, raw load
            ### - subset
            print("starting work on aws instance", i)
            if i not in aws_skip_begin:
                print("badda bing", i)
                """print('ssh -i "' + aws_pems_dir + '" ' + aws_workers[i] +
                          ' "cd ~/' + aws_inst_directory +
                          '; nohup ~/anaconda3/bin/python align_raster.py core_aws=false core_aw2=true ' +
                          'locl_ras=../' + aws_inst_directory +
                          '/ locl_tro=./geotifs_raw locl_rjo=geotifs_raw/ locl_clp=aligned_out skip_trm=true ' +
                          'skip_rrj=true skip_erj=true skip_res=true skip_raw=true pram_sub=' + assignment_str[i][:-1] +
                          ' > ~/align_log.txt"')"""
                subprocess.Popen(["zsh", "aws_psychic.sh", str(aws_pems_dir), str(aws_workers[i]), str(aws_inst_directory), str(assignment_str[i][:-1]), str(i)])
            else:
                print("skip aws start")

        print("closing local work. resampling continuing on aws instances.")
        time.sleep(15)
        sys.exit(0)

    if multiprocessing_mode:
        print("available cpus", multiprocessing.cpu_count())
        print("xr_npar", xr_nparray[0].shape)

    driver = gdal.GetDriverByName("GTiff")

    ### deal with multiprocessing
    ### iterate over layers to align
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
        print(layer_geotif.shape)

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

        print("saved layer", data_info[i]["name"])

        ### now, if in aws, send files to s3
        if aws_work_mode:
            ### move reprojected layers
            print("moving reprojected layer to aws s3")
            os.system('aws s3 cp ' + reproj_step_out + data_info[i]["name"] + '.tif ' + s3_reproj_loc + data_info[i]["name"] + '.tif')
            ### move aligned layer
            print("moving realigned layer to aws s3")
            os.system('aws s3 cp ' + outname + ' ' + s3_aligned_loc + data_info[i]["name"] + '.tif')
