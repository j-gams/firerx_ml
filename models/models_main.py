### IMPORTS

### setup
### send to train...
### compile and save data
import time
from modelbase import mltools as mlt
from modelbase.cascade_late_a import model_cascade2_late_a
from modelbase.cascade_mid import model_cascade2_mid
from modelbase.cascade_early import model_cascade2_early
from modelbase.cascade_early_b import model_cascade2_early_b
from modelbase.vit_pyrencoder import multi_vit
from modelbase.cube_cnn import model_flat2

import os
from data_handler import data_wrangler

import sys

def get_model(modeltype):
    ### INITIALIZE CORRECT MODEL TYPE
    ###
    if modeltype == "vit":
        return multi_vit()
    if modeltype == "f2_baseline":
        return model_flat2()
    if modeltype == "c2_early":
        return model_cascade2_early()
    if modeltype == "c2_early_b":
        return model_cascade2_early_b()
    if modeltype == "c2_mid":
        return model_cascade2_mid()
    if modeltype == "c2_late_a":
        return model_cascade2_late_a()
    if modeltype == "c2_late_b":
        return None

def setup_train_model(actions, run_params, model_params, metadata):
    ### computed metrics ... format should be [action][fold][metric][granularity][ylayer]
    ### after reformat ... [action][metric_type+granularity][ylayer][fold]
    computed_metrics = {}


    ### iterate over train, val, test...
    ### conflict between folds in train, vali vs experiments that dont have folds...
    ### think I need to do this by action and not by ...
    for action in actions:
        ### SECTION 1: SETUP WORK FOR THIS ACTION
        wranglers = {}
        action_modes = run_params["train_params"]["mode"][action]
        for i in range(len(action_modes)):
            ### initialize wrangler
            sw_temp = False
            if action_modes[i] == "train" or action_modes[i] == "combine":
                sw_temp = True
            wranglers[action_modes[i]] = data_wrangler(run_params["data_root_dir"], metadata["n_layers"], run_params["train_params"]["run_on_folds"], 
                                         metadata["layer_dims"], run_params["train_params"]["batch_size"], metadata["buffer_nodata"], metadata["x_layers"],
                                         metadata["y_layers"], sample_weights=sw_temp, low_mem=run_params["data_low_mem"])
            ### setup wrangler
            wranglers[action_modes[i]].setup(action_modes[i])

        if action == "train" or action == "resume":
            computed_metrics[action] = []
            print("TRAINING")
            ### compute sample weights -- this is over all folds, etc
            sw = mlt.compile_sample_weights("bin_log", model_params["hyperparams"]["sw_compute"], action, wranglers[action_modes[0]], model_params["hyperparams"]["sw_bins"], 
                                            run_params["train_params"]["run_on_folds"], run_params["data_root_dir"],
                                            make_vis=True, set_weights=False)
            ### set sample weights for each 
            for i in range(len(action_modes)):
                wranglers[action_modes[i]].set_sample_weights(sw)
            for fold in run_params["train_params"]["run_on_folds"]:
                ### set start time
                action_start_time = time.process_time()
                if run_params["train_params"]["verbosity"] != 0:
                    print(" *** beginning", action, "fold", fold)

                ### set wrangler fold
                wranglers[action_modes[0]].set_fold(fold)
                wranglers[action_modes[1]].set_fold(fold)
                
                ### INITIALIZE MODEL
                working_model = get_model(run_params["model_type"])
                working_name = run_params["model_name"] + "_" + str(fold)
                working_model.setup(model_params["hyperparams"], metadata, run_params["model_dir"], working_name,
                                    run_params["train_params"]["verbosity"], run_params["train_params"]["callbacks"])

                ### LOAD MODEL IF WE ARE RESUMING
                if action == "resume":
                    working_model.load()
                    if run_params["train_params"]["verbosity"] != 0:
                        print("loaded model")
                
                ### TRAIN MODEL FOR N EPOCHS
                print("training model for", run_params["train_params"]["n_epochs_default"], "epochs")
                working_model.fit(wranglers[action_modes[0]], wranglers[action_modes[1]], 
                                  run_params["train_params"]["n_epochs_default"], 
                                  run_params["train_params"]["workers"], run_params["train_params"]["multip"])
                if run_params["train_params"]["verbosity"] != 0:
                    print("done fitting model")

                ### SAVE MODEL
                if run_params["train_params"]["save_model"]:
                    working_model.save()
                
                time_delta = time.process_time() - action_start_time
                computed_metrics[action].append({"time": time_delta})

            del wranglers[action_modes[0]].h5_data
            del wranglers[action_modes[1]].h5_data

        if action == "val":
            computed_metrics[action] = []
            print("VALIDATING")
            for i in range(len(action_modes)):
                wranglers[action_modes[i]].set_sample_weights(sw)
            for fold in run_params["train_params"]["run_on_folds"]:
                ### set start time
                action_start_time = time.process_time()
                ### set wrangler fold
                wranglers[action_modes[0]].set_fold(fold)
                ### INITIALIZE MODEL
                working_model = get_model(run_params["model_type"])
                working_name = run_params["model_name"] + "_" + str(fold)
                working_model.setup(model_params["hyperparams"], metadata, run_params["model_dir"], working_name,
                                    run_params["train_params"]["verbosity"], run_params["train_params"]["callbacks"])
                ### LOAD MODEL
                working_model.load()
                ### COMPUTE METRICS
                computed_metrics[action].append(mlt.compute_metrics(working_model, wranglers[action_modes[0]], 
                                                      run_params["train_params"]["metrics_params"], make_plts=True))
                computed_metrics[action][-1]["time"] =  time.process_time() - action_start_time

            del wranglers[action_modes[0]].h5_data

        if action == "test":
            computed_metrics[action] = []
            print("TESTING")
            for i in range(len(action_modes)):
                wranglers[action_modes[i]].set_sample_weights(sw)
            ### INITIALIZE MODEL
            working_model = get_model(run_params["model_type"])
            working_name = run_params["model_name"] + "_" + str(0)
            working_model.setup(model_params["hyperparams"], metadata, run_params["model_dir"], working_name,
                                    run_params["train_params"]["verbosity"], run_params["train_params"]["callbacks"])
            ### LOAD MODEL
            working_model.load()
            ### COMPUTE METRICS
            computed_metrics[action].append(mlt.compute_metrics(working_model, wranglers[action_modes[0]], 
                                                    run_params["train_params"]["metrics_params"], make_plts=True))
            computed_metrics[action][-1]["time"] =  time.process_time() - action_start_time

            del wranglers[action_modes[0]].h5_data

    ### compile metrics for this action
    print("reformatting metrics...")
    ref_metrics = mlt.reformat_2(actions, computed_metrics, run_params["train_params"]["metrics_params"], metadata["y_layers"])
    ref_metrics["metafolds"] = run_params["train_params"]["run_on_folds"]
    mlt.save_metrics(ref_metrics, run_params["model_dir"])

    
