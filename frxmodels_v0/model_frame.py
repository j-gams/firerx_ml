### model frame
### idea: load datasets
### train models
import sys
import json
#sys.path.append("../utils")
#import utils
import models_main as ml
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
frame_models_configs_locs = frame_config["core"]["model_dicts_locs"]
frame_verbosity = frame_config["core"]["verbosity"]
frame_override_existing_dir = frame_config["core"]["override_existing_dir"]
frame_override_epochs = frame_config["core"]["override_epochs"]
frame_override_remove_ids = frame_config["core"]["remove_ids"]

print(frame_models_configs_locs)

sample_weights_flag = False
sample_weights_carry = None

for i in range(len(frame_models_configs_locs)):
    ### OVERVIEW
    ### - load config
    ### - directory setup
    ### - load metadata
    ### - initialize wranglers
    ### - setup_train_model with parworkers=train_params["workers"]ams

    models_configs_locs = frame_models_configs_locs[i]
    print("running model", i, "of", len(frame_models_configs_locs), "from", models_configs_locs)
    model_config_i = read_config(models_configs_locs)
    ### load model params from config file

    ### visualization_params
    make_vis = model_config_i["run_params"]["make_vis"]

    ### data
    data_root_dir = model_config_i["run_params"]["data_root_dir"]
    data_low_mem = model_config_i["run_params"]["data_low_mem"]

    ### run params
    train_params = model_config_i["run_params"]["train_params"]

    ### model params
    model_dir = model_config_i["model_params"]["model_dir"]
    model_name = model_config_i["model_params"]["model_name"]
    model_parameters = model_config_i["model_params"]

    ### directory setup
    if not frame_override_existing_dir and os.path.exists(model_dir) and train_params["mode"] == "train":
        print("model already exists!")
        print("exiting...")
        sys.exit(0)
    elif frame_override_existing_dir and os.path.exists(model_dir):
        if train_params["mode"] == "train":
            os.system("rm " + model_dir + "/*")
    elif train_params["mode"] == "loadtrain" or train_params["mode"] == "load":
        print("loading existing model...")
    elif train_params["mode"] == "test_1":
        print("testing existing model (1) ...")
    else:
        os.system("mkdir " + model_dir)

    ### TODO -- move config to folder
    os.system("cp " + models_configs_locs + " " + model_dir + "/")

    ### load metadata
    metadata_cols = []
    with open(data_root_dir + "info.txt", 'r') as infofile:
        metadata_total = infofile.read().replace('\n', ';')
    metadata_lines = metadata_total.split(";")
    metadata = []
    for metal in metadata_lines:
        if len(metal) > 0:
            metadata.append(metal.split(","))

    other_info = metadata.pop(0)
    other_info = {"n_folds": other_info[0], "buffer_nodata": other_info[1], "base_crs": other_info[2],
                  }
    n_layers = len(metadata)
    layer_info = {"layer_dims": [], "x_layers": [], "y_layers": [], "layer_names": []}

    for j in range(n_layers):
        layer_info["layer_dims"].append(int(metadata[j][0]))
        layer_info["layer_names"].append(metadata[j][3])
        if metadata[j][1] == "x":
            layer_info["x_layers"].append(j)
        else:
            layer_info["y_layers"].append(j)

    print("metadata check")
    for elt in metadata:
        print(elt)

    
    if frame_override_epochs != -1:
        print("overriding to", frame_override_epochs, "epochs")
        train_params["n_epochs_default"] = frame_override_epochs

    ### setup data wranglers
    train_wrangler = data_wrangler(data_root_dir, n_layers, len(train_params["run_on_folds"]), layer_info["layer_dims"],
                                   train_params["batch_size"], other_info["buffer_nodata"], layer_info["x_layers"],
                                   layer_info["y_layers"], sample_weights=True, low_mem=data_low_mem)
    val_wrangler = data_wrangler(data_root_dir, n_layers, len(train_params["run_on_folds"]), layer_info["layer_dims"],
                                 train_params["batch_size"], other_info["buffer_nodata"], layer_info["x_layers"],
                                 layer_info["y_layers"], sample_weights=True, low_mem=data_low_mem)
    if not sample_weights_flag:
        train_wrangler.compute_sample_weight("real_bininv", nbins=200, vis=True)
        sample_weights_carry = train_wrangler.sample_weights
    else:
        train_wrangler.set_sample_weights(sample_weights_carry)
    val_wrangler.set_sample_weights(sample_weights_carry)
    train_wrangler.exclude_ids(frame_override_remove_ids)
    val_wrangler.exclude_ids(frame_override_remove_ids)
    layer_info["x_layers"] = train_wrangler.x_ids
    layer_info["layer_dims"] = [layer_info["layer_dims"][ii] for ii in layer_info["x_layers"] + layer_info["y_layers"]]
    layer_info["layer_names"] = [layer_info["layer_names"][ii] for ii in layer_info["x_layers"] + layer_info["y_layers"]]
    model_parameters["hyperparams"] = model_parameters["hyperparams"] | layer_info

    ml.setup_train_model(train_params, train_wrangler, val_wrangler, model_parameters)
    #, use_multiprocessing=True, workers=0

    del train_wrangler.h5_data
    del val_wrangler.h5_data



