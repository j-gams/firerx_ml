### DO IMPORTS
### name == main necessary for multiprocessing
if __name__ == "__main__":
    print("init - importing...")
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
    import trim_reproj_raster as rtools
    from utils import utils
    import subprocess
    import time

    ### DEAL WITH PARAMETERS, CONFIG FILES...
    config_dir = "configs/"
    config_prefix = "dbci_"
    config_name = "default_config"
    if len(sys.argv) > 2 and sys.argv[2][:6] == "config":
        config_name = sys.argv[2][7:]
    config_loc = config_dir + config_prefix + config_name + ".json"
    ### load config
    config = utils.read_config(config_loc)
    ### distribute config values to parameters...
    print("init - loaded config file")

    ### Parameters from config file...
    ### whether to multiprocess, allowing parallelization of alignment step
    ### this is highly recommended to speed up the process
    multiprocessing_mode = config["core"]["multiprocessing_mode"]

    raster_src = config["name"]["raster_src"]
    ### put steps under the umbrella of the collection
    collection_name = config["name"]["collection_name"]
    collection_prefix = config["name"]["collection_directory_prefix"] + collection_name
    ### step 1 save (trim to CO)
    trim_step_out = config["name"]["trim_step_out"]
    ### step 2 reproject (WGS 84)
    reproj_step_out = config["name"]["reproj_step_out"]
    ### step 3 re-align
    align_step_out = config["name"]["align_step_out"]
    ### make more useful path variables from parameters
    trim_step_dir = collection_prefix + trim_step_out
    reproj_step_dir = collection_prefix + reproj_step_out
    align_step_dir = collection_prefix + align_step_out

    ### parameters
    ### overlap to provide subsets (to prevent pixels for being cut out for intersecting boundary)
    subdiv_meter_overlap = config["params"]["subdiv_meter_overlap"]
    ### slop --- extra safety margin on all sides, where the overlap is only on 2
    subdiv_meter_slop = config["params"]["subdiv_meter_slop"]
    ### how to cut the extent... (horizontal blocks, vertical blocks)
    subdiv_hv_split = config["params"]["subdiv_hv_split"]

    res_check_tol = config["params"]["res_check_tol"]

    output_nodata = config["params"]["output_nodata"]

    resampling_mode = config["params"]["resampling_mode"]


    ### whether to skip loading the guiding layer first (needed for some other early steps
    skip_guiding_load = config["skip"]["guiding_load"]
    ### whether to skip subdividing the extent (batching the inputs)
    skip_subdivision = config["skip"]["subdivision"]
    ### whether to skip trimming layers to the extent
    skip_trim = config["skip"]["trim"]
    ### whether to skip reprojecting the extent file before trimming rasters
    skip_extent_reproj = config["skip"]["extent_reproj"]
    ###
    skip_raster_reproj = config["skip"]["raster_reproj"]
    ### same --
    skip_resolution_check = config["skip"]["resolution_check"]
    ### same --
    skip_alignment = config["skip"]["alignment"]
    ### same --

    ### same --
    guiding_layer_idx = config["data"]["guiding_layer_idx"]
    data_info = config["data"]["data_info"]
    ### extent source with different formulation now
    extent_info = config["data"]["extent_info"]
    extent_epsg_override = config["data"]["extent_epsg_override"]
    exclude = config["data"]["exclude"]

    ### diagnostic check on what we're doing here...
    print("init - process summary:")
    print("  - skipping guiding load:", skip_guiding_load)
    print("  - skipping trim:", skip_trim)
    print("  - skipping raster reproj:", skip_raster_reproj)
    print("  - skipping vector reproj:", skip_extent_reproj)
    print("  - skipping layer resolution check:", skip_extent_reproj)
    print("  - skipping layer alignment:", skip_alignment)
    print("  - running in multiprocessing mode:", multiprocessing_mode)
    print("  - resampling with ", resampling_mode)

    ### Set up directory structure
    ### set up environment -- make collection directory
    ### TODO -- make sure we have subfolders in each of these for subregions...?
    os.system("mkdir " + collection_prefix)
    ### make trim step working directory
    os.system("mkdir " + trim_step_dir)
    ### make reproj step working directory
    os.system("mkdir " + reproj_step_dir)
    ### make output (aligned) directory
    os.system("mkdir " + align_step_dir)

    ### makes gdal less annoying
    gdal.UseExceptions()

    print("init - subsetting dataset")
    subset = []
    init_guiding_idx = guiding_layer_idx
    for i in range(len(data_info)):
        if i in exclude:
            if i < guiding_layer_idx:
                init_guiding_idx -= 1
        else:
            subset.append(data_info[i])
    guiding_layer_idx = init_guiding_idx
    data_info = subset


    ### TODO -- rework data_info to be more like create_pyramid_set -- everything thorugh data_info with guiding idx

    ### LOAD RAW GUIDING LAYER DATA ... for setup. delete after.
    if not skip_guiding_load:
        print("init - loading guiding layer: " + data_info[guiding_layer_idx]["loc"])
        temp_guiding_raster = gdal.Open(raster_src + data_info[guiding_layer_idx]["loc"])
        temp_guiding_band = temp_guiding_raster.GetRasterBand(1)
        temp_guiding_ndval = temp_guiding_band.GetNoDataValue()
        temp_guiding_size = (temp_guiding_raster.RasterXSize, temp_guiding_raster.RasterYSize)
        tgulh, tgph, _, tgulv, _, tgpv = temp_guiding_raster.GetGeoTransform()
        tgpv = abs(tgpv)
        temp_guiding_proj = temp_guiding_raster.GetProjection()
        temp_guiding_crs = (tgulh, tgulv, tgph, tgpv)
        temp_guiding_np = temp_guiding_raster.ReadAsArray().transpose()
        temp_guiding_geotrans = temp_guiding_raster.GetGeoTransform()
        temp_guiding_epsg = osr.SpatialReference(wkt=temp_guiding_proj).GetAttrValue('AUTHORITY', 1)
        print("init - guiding layer epsg", temp_guiding_epsg)

    ### LOAD EXTENT ... but only if not skip_trim ... if we are skipping trim we dont have to deal with the extent
    if not skip_trim:
        ### STEP 1 -- need to load extent and reproject it to the base epsg before continuing
        ### STEP 2 -- need to subdivide extent
        ### STEP 3 -- need to iterate over subregions and reproj raster -> trim layer -> reproj layer.
        print("extent - loading extent")
        extent = gdal.OpenEx(extent_info["extent_src"] + extent_info["extent_name"] + ".shp",
                             sibling_files=[extent_info["extent_src"] + extent_info["extent_name"] + ".prj"])
        print("")
        extent_proj = extent.GetLayer().GetSpatialRef()
        extent_epsg = extent_proj.GetAttrValue('AUTHORITY', 1)
        ### if there is trouble extracting the genuine projection and we are getting nonsense, reset it to the known val
        if extent_epsg_override is not False:
            print("extent - overriding extent epsg from", extent_epsg, "to", extent_epsg_override)
            extent_epsg = int(extent_epsg_override)
        print("extent - epsg check", extent_epsg)
        ### now reproject the base extent to match the projection of the guiding raster layer
        if not skip_extent_reproj:
            print("extent - reprojecting and saving extent shapefile in original directory")
            ### this reprojects the base extent to the guiding layer epsg
            #os.system('ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:' + str(temp_guiding_epsg) + ' -s_srs EPSG:' +
            #          str(extent_epsg) + ' ' + extent_info["extent_src"] + extent_info["extent_name"] + '_' +
            #          str(temp_guiding_epsg) + '.shp ' + extent_info["extent_src"] + extent_info["extent_name"] + '.shp')
            ### this is the guiding-proj version of the file to be used
            correct_proj_extent_name = extent_info["extent_name"] + '_' + str(temp_guiding_epsg)
            ### this is the old version to be used as a starting point for reprojecting the extent only
            new_proj_base = extent_info["extent_name"]
        else:
            ### here, because we are fine, the guiding-proj version is the same as reprojection base
            correct_proj_extent_name = extent_info["extent_name"]
            new_proj_base = extent_info["extent_name"]

    ### do subdivision here
    if subdiv_hv_split is None:
        subdiv_hv_split = (1, 1)
    subregions = []
    for i in range(subdiv_hv_split[0]):
        for j in range(subdiv_hv_split[1]):
            subregions.append((i, j))
    if not skip_subdivision:
        ### perform some calculations for the subdivision step...
        subdiv_geo_overlap = (subdiv_meter_overlap/data_info[guiding_layer_idx]["base_res"]) * tgph
        subdiv_geo_slop = (subdiv_meter_slop/data_info[guiding_layer_idx]["base_res"]) * tgph

        rtools.subdivide_extent(extent_info["extent_src"], extent_info["extent_name"],# + '_' + str(extent_epsg),
                                subdiv_hv_split, subdiv_geo_overlap, subdiv_geo_slop, extent_epsg)

    ### NOW POST-SUBDIV...
    ### now... need to iterate over subregions
    for subregion_idx in range(len(subregions)):
        sub_i, sub_j = subregions[subregion_idx]

        sub_datsrc = ogr.Open(extent_info["extent_src"] + "subregions/" + new_proj_base + '_' + str(extent_epsg) + '_' + str(sub_i) + '_' + str(sub_j) + '.shp')
        numLayers = sub_datsrc.GetLayerCount()
        sub_vlayer = sub_datsrc.GetLayer()
        sub_area = 0
        for vfeature in sub_vlayer:
            vgeom = vfeature.GetGeometryRef()
            sub_area = vgeom.GetArea()

        print("subregion area", (sub_i, sub_j), sub_area)
        if sub_area == 0:
            print("subregion with 0 area:", (sub_i, sub_j))
            continue

        ### load unaligned raw data
        raster = []
        raster_proj = []
        raster_ndvals = []
        raster_size = []
        raster_crs = []
        raster_nparray = []

        ### LOAD RAW raster DATA
        print("raw - loading raw raster data")
        for i in range(len(data_info)):
            """if i == guiding_layer_idx:
                continue"""
            print("  - loading raw " + data_info[i]["loc"] + " data (" + str(i) + ")...")
            temp_raster = gdal.Open(raster_src + data_info[i]["loc"])
            temp_band = temp_raster.GetRasterBand(1)
            temp_proj = temp_raster.GetProjection()
            temp_ndval = temp_band.GetNoDataValue()
            _, temp_hres, _, _, _, temp_vres = temp_raster.GetGeoTransform()
            ### IF TRIM --- TRIM TO subregion AOI!
            if not skip_trim:
                ### determine projection for shapefile... need to reproject extent for easier raster trim/reproj
                ### below 2 lines -- get epsg number from this raster layer
                wktproj = osr.SpatialReference(wkt=temp_proj)
                layer_epsg = wktproj.GetAttrValue('AUTHORITY', 1)
                print("trim - layer projection is EPSG:" + str(layer_epsg))
                ### deal with layers woth messed up metadata...
                if data_info[i]["override_input_proj"] is not False:
                    print("  - overriding input layer projection to EPSG:" + str(data_info[i]["override_input_proj"]))
                    layer_epsg = str(data_info[i]["override_input_proj"])
                ### reproject extent to current layer raw projection
                ### reproject extent to current layer raw projection
                os.system('ogr2ogr -f "ESRI Shapefile" -t_srs EPSG:' + str(layer_epsg) + ' -s_srs EPSG:' +
                          str(extent_epsg) + ' ' + extent_info["extent_src"] + "subregions/" + new_proj_base + '_' +
                          str(layer_epsg) + '_' + str(sub_i) + '_' + str(sub_j) + '.shp ' + extent_info["extent_src"] +
                          "subregions/" + extent_info["extent_name"] + '_' + str(extent_epsg) + '_' + str(sub_i) + '_' +
                          str(sub_j) + '.shp')
                print("  - reprojected shapefile to raw layer proj: EPSG", layer_epsg)
                ### trim to extent in raw projection
                os.system('gdalwarp -cutline ' + extent_info["extent_src"] + "subregions/" + new_proj_base + '_' +
                          str(layer_epsg) + '_' + str(sub_i) + '_' + str(sub_j) + '.shp -crop_to_cutline -dstnodata "' +
                          str(temp_ndval) + '" ' + raster_src + data_info[i]["loc"] + ' ' + trim_step_dir +
                          data_info[i]["name"] + '_' + str(temp_guiding_epsg) + '_' + str(sub_i) + '_' + str(sub_j) + ".tif")
                print("  - saved trimmed file")

            ### IF PROJECTION IS NOT BASE PROJECTION --- NEED TO REPROJECT!!!!!
            if not skip_raster_reproj:
                if layer_epsg == temp_guiding_epsg:
                    print("  - no reprojection required -- x epsg =",layer_epsg," -- guiding epsg =", temp_guiding_epsg)
                    ### check whether reported resolution is consistent with actual pixel widths
                    if not skip_resolution_check and i != guiding_layer_idx:
                        print("  - verifying reported layer resolution")
                        ### reported pixel widths -- base * fraction of base reported to layer reported
                        reported_hres = tgph * (data_info[i]["base_res"] / data_info[guiding_layer_idx]["base_res"])
                        reported_vres = tgpv * (data_info[i]["base_res"] / data_info[guiding_layer_idx]["base_res"])
                        actual_hres = temp_hres
                        actual_vres = temp_vres
                        if abs(reported_hres - actual_hres) > res_check_tol or abs(reported_vres -
                                                                                   actual_vres) > res_check_tol:
                            ### this fails the resolution tolerance check
                            print("  - layer resolution outside of tolerance -- reported", reported_hres, reported_vres,
                                  "-- actual --", actual_hres, actual_vres)
                            ### need to resample to reported resolutions
                            gdal.Warp(reproj_step_dir + data_info[i]["name"] + '_' + str(layer_epsg) + '_' + str(sub_i)+
                                      '_' + str(sub_j) + ".tif", trim_step_dir + data_info[i]["name"] + '_' +
                                      str(layer_epsg) + '_' + str(sub_i) + '_' + str(sub_j) +
                                      ".tif", xRes=reported_hres, yRes=reported_vres)
                        else:
                            ### passes tolerance check -- no action required
                            print("  - layer resolution within tolerance -- reported =", reported_hres, reported_vres,
                                  " -- actual =", actual_hres, actual_vres)
                            os.system('cp ' + trim_step_dir + data_info[i]["name"] + '_' + str(layer_epsg) + '_' +
                                      str(sub_i) + '_' + str(sub_j) + '.tif ' + reproj_step_dir + data_info[i]["name"] +
                                      '_' + str(layer_epsg) + '_' + str(sub_i) + '_' + str(sub_j) + '.tif')
                    else:
                        ### move the file along to the next checkpoint...
                        os.system('cp ' + trim_step_dir + data_info[i]["name"] + '_' + str(layer_epsg) + '_' +
                                  str(sub_i) + '_' + str(sub_j) + '.tif ' + reproj_step_dir + data_info[i]["name"] +
                                  '_' + str(layer_epsg) + '_' + str(sub_i) + '_' + str(sub_j) + '.tif')
                else:
                    print("  - reprojection required -- x epsg = ", layer_epsg, "-- guiding epsg = ", temp_guiding_epsg)
                    ### compute desired pixel resolutions in consistent crs by computing ratio
                    hres_frac = tgph * (data_info[i]["base_res"] / data_info[guiding_layer_idx]["base_res"])
                    vres_frac = tgpv * (data_info[i]["base_res"] / data_info[guiding_layer_idx]["base_res"])
                    ### reproject to guide projection and resample at appropriate resolution
                    gdal.Warp(reproj_step_dir + data_info[i]["name"] + '_' + str(temp_guiding_epsg) + '_' + str(sub_i) +
                              '_' + str(sub_j) + '.tif',
                              trim_step_dir + data_info[i]["name"] + '_' + str(temp_guiding_epsg) + '_' + str(sub_i) + '_' +
                              str(sub_j) + '.tif',
                              xRes=hres_frac, yRes=vres_frac, dstSRS='EPSG:' + temp_guiding_epsg)
                    print("  - saved reprojected layer")

            if not skip_alignment:
                ### reload fixed files
                ### Trimming and reprojection are complete
                ### load trimmed and reprojected rasters
                print("  - loading fixed file -", end="", flush=True)
                ### open file with gdal
                raster.append(gdal.Open(reproj_step_dir + data_info[i]["name"] + '_' + str(temp_guiding_epsg) + '_' + str(sub_i) + '_' + str(sub_j) +
                                        '.tif'))
                ### get data from first raster band
                tdata_rband = raster[-1].GetRasterBand(1)
                ### record projection
                raster_proj.append(raster[-1].GetProjection())
                ### get nodata value from raster band
                raster_ndvals.append(tdata_rband.GetNoDataValue())
                ### get raster grid size
                raster_size.append((raster[-1].RasterXSize, raster[-1].RasterYSize))
                ### get crs parameters from raster
                ulh, ph, _, ulv, _, pv = raster[-1].GetGeoTransform()
                pv = abs(pv)
                ### record simple crs
                raster_crs.append((ulh, ulv, ph, pv))
                ### extract np array from raster
                raster_nparray.append(raster[-1].ReadAsArray().transpose())
                print("-> done")
        if not skip_alignment:
            if multiprocessing_mode:
                print("multiprocessing - available cpus", multiprocessing.cpu_count())
                print("multiprocessing - first raster shape", raster_nparray[0].shape)

            driver = gdal.GetDriverByName("GTiff")
            ### deal with multiprocessing
            ### iterate over layers to align
            print("processing layers...")
            for i in range(len(data_info)):
                ### basic resampling parameters
                if i == guiding_layer_idx:
                    continue
                guiding_layer_res = data_info[guiding_layer_idx]["base_res"]
                layer_original_res = data_info[i]["base_res"]
                layer_output_res = data_info[i]["output_res"]
                layer_sampling_res = data_info[i]["resample_res"]
                layer_original_nodata = raster_ndvals[i]
                layer_realign_nodata = output_nodata
                layer_avg_func = data_info[i]["avg_func"]

                ###compute total size of grid in meters
                total_x_meters = data_info[guiding_layer_idx]["base_res"] * raster_size[guiding_layer_idx][0]
                total_y_meters = data_info[guiding_layer_idx]["base_res"] * raster_size[guiding_layer_idx][1]

                ### compute size of output from desired resolution and total meters
                output_x_size = math.ceil(total_x_meters / data_info[i]["output_res"])
                output_y_size = math.ceil(total_y_meters / data_info[i]["output_res"])

                ### compute sampling size... target res // resample_res
                sample_grid_size = math.ceil(data_info[i]["output_res"] / data_info[i]["resample_res"])

                ### target crs -- same ulh, ulv as y crs, widths defined by desired dim
                target_crs = [tgulh, tgulv, (data_info[i]["output_res"] / data_info[guiding_layer_idx]["base_res"] * tgph),
                              (data_info[i]["output_res"] / data_info[guiding_layer_idx]["base_res"] * tgpv)]

                # slice width, sample_grid_size, layer_crs, target_crs, layer_nodata_val, oob_nodata_val, avg_method
                resample_params = [output_y_size, sample_grid_size, raster_crs[i], target_crs, raster_ndvals[i], output_nodata,
                                   data_info[i]["avg_method"], resampling_mode]
                map_params = []
                print(output_x_size)
                if multiprocessing_mode:
                    ### determine number of cpus available for multiprocessing purposes
                    ncpu = multiprocessing.cpu_count()
                else:
                    ncpu = 1
                for j in range(ncpu):
                    map_params.append([align.chunk_list(output_x_size, ncpu, j), raster_nparray[i], resample_params])

                ## generate feed parameters
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
                outname = align_step_dir + "/" + data_info[i]["name"] + "_" + str(sub_i) + "_" + str(sub_j) + ".tif"
                layer_out = driver.Create(outname, layer_geotif.shape[0], layer_geotif.shape[1], 1,
                                          gdal.GDT_Float32)
                layer_out.SetGeoTransform(layer_full_crs)
                layer_out.SetProjection(raster_proj[guiding_layer_idx])

                ### save layer geotif
                layer_out.GetRasterBand(1).WriteArray(layer_geotif.transpose())
                layer_out.GetRasterBand(1).SetNoDataValue(output_nodata)
                layer_out.FlushCache()

                print("  - saved layer", data_info[i]["name"])

        print("completed work for subregion", subregion_idx)
    print("all done")

