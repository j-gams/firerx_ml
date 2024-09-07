### IMPORTS

### setup
### send to train...
### compile and save data
import time
from modelbase import mltools as mlt
from modelbase.cascade_late_a import model_cascade2_late_a
from modelbase.cascade_mid import model_cascade2_mid
from modelbase.cascade_early import model_cascade2_early
from modelbase.vit_pyrencoder import multi_vit
from modelbase.cube_cnn import model_flat2

import os
from data_handler import data_wrangler

import sys

def setup_train_model(train_params, train_wrangler, val_wrangler, model_params):
    ### make and setup models...
    setup_start_time = time.process_time()

    ### set wrangler mode
    train_wrangler.set_mode("train")
    val_wrangler.set_mode("val")

    computed_metrics = []

    ### train models if required:
    for fold in train_params["run_on_folds"]:
        if train_params["verbosity"] != 0:
            print(" *** beginning fold", fold)

        train_wrangler.set_fold(fold)
        val_wrangler.set_fold(fold)

        ### INITIALIZE CORRECT MODEL TYPE
        ###
        if model_params["model_type"] == "vit":
            working_model = multi_vit()
        if model_params["model_type"] == "f2_baseline":
            working_model = model_flat2()
        if model_params["model_type"] == "c2_early":
            working_model = model_cascade2_early()
        if model_params["model_type"] == "c2_mid":
            working_model = model_cascade2_mid()
        if model_params["model_type"] == "c2_late_a":
            working_model = model_cascade2_late_a()
        if model_params["model_type"] == "c2_late_b":
            working_model = None

        working_model.setup(model_params["hyperparams"], model_params["model_dir"], model_params["model_name"],
                            train_params["verbosity"], train_params["callbacks"])

        ### LOAD MODEL IF REQUIRED
        ###
        if train_params["mode"] == "load" or train_params["mode"] == "loadtrain":
            working_model.load()

        if train_params["mode"] == "loadtrain" or train_params["mode"] == "train":
            print("training model for", train_params["n_epochs_default"], "epochs")
            working_model.fit(train_wrangler, val_wrangler, train_params["n_epochs_default"],
                              train_params["workers"], train_params["multip"])
            if train_params["verbosity"] != 0:
                print("done fitting model")
            if train_params["save_model"]:
                working_model.save()
        if train_params["compute_metrics"]:
            computed_metrics.append(mlt.compute_metrics(working_model, val_wrangler, train_params["metrics_params"]))
       #print(working_model.callbacks[0].logs["loss"])
    time_delta = time.process_time() - setup_start_time
    print("reformatting metrics...")
    ref_metrics = mlt.reformat_metrics(computed_metrics, train_params["metrics_params"], model_params["hyperparams"]["y_layers"])
    ref_metrics["metafolds"] = train_params["run_on_folds"]
    ref_metrics["total_time"] = time_delta
    mlt.save_metrics(ref_metrics, working_model.modeldir)
