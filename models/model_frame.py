### model frame
### idea: load datasets
### train models
import sys
import json
#sys.path.append("../utils")
#import utils
import models_main as ml
import modelbase.mltools as mlt
import os
from data_handler import data_wrangler

def read_config(in_loc):
    config_in = open(in_loc)
    config_dict = json.load(config_in)
    return config_dict

config_dir = "configs/"
config_prefix = "mlframe_"
config_name = "default_config"
if len(sys.argv) < 2:
    print("using default model_frame config")
elif sys.argv[2][:6] == "config":
    config_name = sys.argv[2][7:]
config_loc = config_dir + config_prefix + config_name + ".json"
frame_config = read_config(config_loc)
print("loaded config file")

### idea -- load config for frame as a whole which specifies which model configs to use
frame_models_configs_locs = frame_config["core"]["models"]
frame_verbosity = frame_config["core"]["verbosity"]
frame_override_existing_dir = frame_config["core"]["override_existing_dir"]
frame_override_epochs = frame_config["core"]["override_epochs"]
frame_override_remove_ids = frame_config["core"]["remove_ids"]

print(frame_config["core"]["models"])

sample_weights_flag = False
sample_weights_carry = None

### report format: actions, status, results
report = []

for i in range(len(frame_config["core"]["models"])):
    report.append({"actions": [],
                   "status": [],
                   "results": []})

    ### OVERVIEW
    ### - load config
    ### - directory setup
    ### - load metadata
    ### - initialize wranglers
    ### - setup_train_model with parworkers=train_params["workers"]ams

    models_configs_locs, actions = frame_config["core"]["models"][i]
    print("running model", i+1, "of", len(frame_config["core"]["models"]), "from", models_configs_locs)
    model_config_i = read_config(models_configs_locs)
    ### load model params from config file

    ### run params
    run_params = model_config_i["run_params"]

    ### model params
    model_parameters = model_config_i["model_params"]

    ### we are training a model from scratch but the directory already exists
    ### we dont want to override so skip
    if not frame_override_existing_dir and os.path.exists(run_params["model_dir"]) and "train" in actions:
        print("model directory already exists!")
        print("skipping", run_params["model_name"])
        report[i]["status"].append("skipped")
        continue
    elif frame_override_existing_dir and os.path.exists(run_params["model_dir"]) and "train" in actions:
        os.system("rm " + run_params["model_dir"] + "/*")
    else:
        os.system("mkdir " + run_params["model_dir"])

    ### copy config to model folder
    os.system("cp " + models_configs_locs + " " + run_params["model_dir"] + "/")

    ### load metadata
    metadata = mlt.load_metadata(run_params["data_root_dir"])

    ### set all models to run with same number of epochs    
    if frame_override_epochs != -1:
        print("overriding to", frame_override_epochs, "epochs")
        run_params["train_params"]["n_epochs_default"] = frame_override_epochs

    ml.setup_train_model(actions, run_params, model_parameters, metadata)
    #, use_multiprocessing=True, workers=0



