import time
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import stats

def compile_sample_weights(method, func_mode, wrangler, n_bins, n_folds, active_fold, save_to, make_vis=False, set_weights=True):
    ### wrangler needs to be set up beforehand...
    sample_weights = [[[] for ii in range(len(wrangler.use_y_ids))] for jj in range(n_folds)]
    if func_mode == "load_from_file":
        print("loading sample weights from", save_to)
        for j in range(len(n_folds)):
            for i in range(len(sample_weights)):
                sample_weights[j][i] = np.load(save_to + "sample_weights_f" + str(j) + "_y" + str(i) + ".csv")
                if j == active_fold and set_weights:
                    wrangler.sample_weights[i] = sample_weights[i]
        return sample_weights
    else:
        ### visualization parameters
        valuefreqs = []
        if make_vis:
            target_names = ["WUE", "ESI", "AGB"]
            colors = ["salmon", "lightgreen", "lightblue"]
        if method == "bin_log":
            print("computing sample weights (bin_log method)...", end="", flush=True)
            ### iterate over y layers and for each, bin and come up with thresholds
            for j in range(len(n_folds)):
                for i in range(len(wrangler.use_y_ids)):
                    ### compute log bin frequencies
                    valuefreq = np.log(np.histogram(wrangler.apply_norm(wrangler.h5_data[wrangler.use_y_ids[i]], wrangler.use_y_ids[i]), bins=n_bins, range=[0, 1])[0])
                    if make_vis:
                        ### make bin frequency plot
                        plt.bar(np.arange(n_bins)/n_bins, valuefreq, width=1/n_bins, color=colors[i])
                        plt.title(target_names[i] + " Log Normalized Pixel Value Frequency (" + str(n_bins) + " bins)")
                        plt.xlabel("Normalized Value")
                        plt.ylabel("Log Bin Frequency")
                        plt.savefig("../visualize/data_dist/sample_freq_plot_" + str(i) + ".png")
                        plt.clf()
                    ### iterate over bins and compute bin weights
                    for j in range(n_bins):
                        if valuefreq[j] == 0:
                            valuefreq[j] = 1
                        else:
                            valuefreq[j] = (wrangler.h5_data[wrangler.use_y_ids[i]].shape[1]**2)/valuefreq[j]
                    ### normalize - scale bin weights to <= 1
                    temp_max = np.max(valuefreq)
                    for j in range(n_bins):
                        valuefreq[j] /= temp_max
                    valuefreqs.append(np.array(valuefreq))
                    if make_vis:
                        ### make bin weight plot
                        plt.bar(np.arange(n_bins)/n_bins, valuefreqs[i], width=1/n_bins, color=colors[i])
                        plt.title(target_names[i] + " Pixel Value Bin Weights (" + str(n_bins) + " bins)")
                        plt.xlabel("Normalized Value (" + str(n_bins) + " bins)")
                        plt.ylabel("Pixel Value Bin Weight")
                        plt.savefig("../visualize/data_dist/sample_weight_plot_" + str(i) + ".png")
                        plt.clf()
                ### values for each class of sample... now map samples to values
                print("halfway (" + str(j) + ")...", end="", flush=True)
                for i in range(len(wrangler.use_y_ids)):
                    for k in range(len(wrangler.h5_data[wrangler.use_y_ids[i]])):
                        ### attrocious line below computes sample weights by summing over sample pixel bin weights
                        wrangler.sample_weights[i].append(np.sum(valuefreqs[i][np.clip((wrangler.apply_norm(wrangler.h5_data[wrangler.use_y_ids[i]][k], wrangler.use_y_ids[i]) * n_bins), None, n_bins-1).astype(int)]))
            print("done")
            if set_weights:
                for i in range(len(wrangler.use_y_ids)):
                    wrangler.sample_weights[i] = np.array(wrangler.sample_weights[i])
        ### save data... 
        print("saving sample weights to", save_to)
        for j in range(len(n_folds)):
            for i in range(len(sample_weights)):
                np.save(save_to + "sample_weights_f" + str(j) + "_y" + str(i) + ".csv", sample_weights[i])
        return sample_weights

class lossCallback(tf.keras.callbacks.Callback):
    def __init__ (self):
        self.logs = dict()
        self.init = False
        self.lasttime = -1

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.process_time()

    def on_epoch_end(self, epoch, logs={}):
        keys = list(logs.keys())
        if not self.init:
            for k in keys:
                self.logs[k] = []
            self.init = True
            self.logs["time"] = []
        #['loss', 'accuracy', 'val_loss', 'val_accuracy']
        #print("CALLBACK: epoch", epoch, "keys:", keys)
        ctime = time.process_time()
        self.logs["time"].append(self.epoch_start_time - ctime)
        for k in keys:
            self.logs[k].append(logs.get(k))

    def resume_from_load(self, loaded_logs):
        self.init = True
        self.logs = loaded_logs

def dense_block(input_layer, dense_layers):
    for dl in dense_layers:
            input_layer = tf.keras.layers.Dense(dl, activation="relu")(input_layer)
    return input_layer

### wue, esi, agb
### 15, 15, 1
def y_block(input_layer, y_dims, y_unique, y_frequency, y_names, block_version, single_task):
    y_split = []
    if block_version == "singletask":
        y_split = tf.keras.layers.Dense((y_dims[single_task] ** 2), activation="relu")(input_layer)
        ysplit2 = tf.keras.layers.Reshape((y_dims[single_task], y_dims[single_task], 1), name=y_names[single_task])(y_split)
        associated_dim = y_dims[single_task]
    else:

        #associated_dim = []
        if block_version == "cascade":
            for j in range(len(y_unique)):
                yj = tf.keras.layers.Dense((y_unique[j] ** 2) * y_frequency[j], activation="relu")(input_layer)
                y_split.append(yj)
            ### hard coded for now...
            ### ecostress branch
            yj2i = tf.keras.layers.Concatenate()(y_split)
            yj2i = tf.keras.layers.Dense(y_dims[0] ** 2 + y_dims[1] ** 2, activation="relu")(yj2i)
            ### wue branch
            yj2a = tf.keras.layers.Dense(y_dims[0] ** 2, activation="relu")(yj2i)
            yj2a = tf.keras.layers.Reshape((y_dims[0], y_dims[0], 1), name=y_names[0])(yj2a)
            ### esi branch
            yj2b = tf.keras.layers.Dense(y_dims[1] ** 2, activation="relu")(yj2i)
            yj2b = tf.keras.layers.Reshape((y_dims[1], y_dims[1], 1), name=y_names[1])(yj2b)

            ### agb branch
            yj2 = tf.keras.layers.Dense((y_dims[2] ** 2), activation="relu")(y_split[1])
            yj2 = tf.keras.layers.Reshape((y_dims[2], y_dims[2], 1), name=y_names[2])(yj2)

            y_final = [yj2a, yj2b, yj2]

        elif block_version == "ultrabasic":
            for j in range(len(y_unique)):
                yj = tf.keras.layers.Dense((y_unique[j] ** 2) * y_frequency[j], activation="relu")(input_layer)
                y_split.append(yj)
            yj2 = tf.keras.layers.Dense((y_unique[1] ** 2) * y_frequency[1], activation="relu")(y_split[0])
            yj2a = tf.keras.layers.Dense((y_dims[0] ** 2), activation="relu")(yj2)
            yj2a = tf.keras.layers.Reshape((y_dims[0], y_dims[0], 1), name=y_names[0])(yj2a)

            yj2b = tf.keras.layers.Dense((y_dims[1] ** 2), activation="relu")(yj2)
            yj2b = tf.keras.layers.Reshape((y_dims[1], y_dims[1], 1), name=y_names[1])(yj2b)

            yj2c = tf.keras.layers.Dense((y_unique[0] ** 2), activation="relu")(y_split[1])
            yj2c = tf.keras.layers.Reshape((y_dims[2], y_dims[2], 1), name=y_names[2])(yj2c)
            y_final = [yj2a, yj2b, yj2c]

        elif block_version == "basic":
            yj = tf.keras.layers.Dense((max(y_unique) ** 2) * 3, activation="relu")(input_layer)

            ### ecostress - wue and esi
            yj2 = tf.keras.layers.Dense((max(y_unique) ** 2) * y_frequency[1], activation="relu")(yj)
            ### wue
            yj2a = tf.keras.layers.Dense((y_dims[0] ** 2), activation="relu")(yj2)
            yj2a = tf.keras.layers.Reshape((y_dims[0], y_dims[0], 1), name=y_names[0])(yj2a)

            ### esi
            yj2b = tf.keras.layers.Dense((y_dims[1] ** 2), activation="relu")(yj2)
            yj2b = tf.keras.layers.Reshape((y_dims[1], y_dims[1], 1), name=y_names[1])(yj2b)

            yj2c = tf.keras.layers.Dense((max(y_unique) ** 2), activation="relu")(yj)
            yj2c = tf.keras.layers.Dense((y_dims[2] ** 2), activation="relu")(yj2c)
            yj2c = tf.keras.layers.Reshape((y_dims[2], y_dims[2], 1), name=y_names[2])(yj2c)
            y_final = [yj2a, yj2b, yj2c]

        elif block_version == "basicnorm":
            yj = tf.keras.layers.Dense((max(y_unique) ** 2) * 3, activation="relu")(input_layer)

            ### ecostress
            yj2 = tf.keras.layers.Dense((max(y_unique) ** 2) * y_frequency[1], activation="relu")(yj)
            ### wue
            yj2a = tf.keras.layers.Dense((y_dims[0] ** 2), activation="relu")(yj2)
            yj2a = tf.keras.layers.Dense((y_dims[0] ** 2), activation="relu", name=y_names[0])(yj2a)

            ### esi
            yj2b = tf.keras.layers.Dense((y_dims[1] ** 2), activation="relu")(yj2)
            yj2b = tf.keras.layers.Dense((y_dims[1] ** 2), activation="relu", name=y_names[1])(yj2b)

            ### agb
            yj2c = tf.keras.layers.Dense((max(y_unique) ** 2), activation="relu")(yj)
            yj2c = tf.keras.layers.Dense((max(y_unique) ** 2), activation="relu")(yj2c)
            yj2c = tf.keras.layers.Dense((y_dims[2] ** 2), activation="relu", name=y_names[2])(yj2c)
            y_final = [yj2a, yj2b, yj2c]

    return y_final

def metric_mse(y_predicted, y_actual, mode, granularity):
    ret = [[] for ii in range(len(granularity))]
    for i in range(len(y_predicted)):
        if mode == "geo":
            mse = (y_predicted[i] - y_actual[i]) ** 2
            for j in range(len(granularity)):
                if granularity[j] == "single":
                    ret[j].append(mse.mean(axis=(1,2)))
                if granularity[j] == "each":
                    ret[j].append(mse)
                if granularity[j] == "overall":
                    ret[j].append(mse.mean())
        elif mode == "flattened":
            mse = (y_predicted[i] - y_actual[i]) ** 2
            for j in range(len(granularity)):
                if granularity[j] == "single":
                    ret[j].append(mse.mean(axis=1))
                if granularity[j] == "each":
                    ret[j].append(mse)
                if granularity[j] == "overall":
                    ret[j].append(mse.mean())
        dimcheck = []
        for elt_j in ret:
            for elt_i in elt_j:
                try:
                    dimcheck.append(elt_i.shape)
                except:
                    dimcheck.append("scalar")
        print("mse dimensions check", dimcheck)
    return ret

def metric_mae(y_predicted, y_actual, mode, granularity):
    ret = [[] for ii in range(len(granularity))]
    for i in range(len(y_predicted)):
        if mode == "geo":
            mae = np.abs(y_predicted[i] - y_actual[i])
            for j in range(len(granularity)):
                if granularity[j] == "single":
                    ret[j].append(mae.mean(axis=(1, 2)))
                if granularity[j] == "each":
                    ret[j].append(mae)
                if granularity[j] == "overall":
                    ret[j].append(mae.mean())
        elif mode == "flattened":
            mae = np.abs(y_predicted[i] - y_actual[i])
            for j in range(len(granularity)):
                if granularity[j] == "single":
                    ret[j].append(mae.mean(axis=1))
                if granularity[j] == "each":
                    ret[j].append(mae)
                if granularity[j] == "overall":
                    ret[j].append(mae.mean())
        dimcheck = []
        for elt_j in ret:
            for elt_i in elt_j:
                try:
                    dimcheck.append(elt_i.shape)
                except:
                    dimcheck.append("scalar")
        print("mse dimensions check", dimcheck)
    return ret

def metric_r2(y_predicted, y_actual, mode, make_plts, model_name):
    r_values = []
    nbins = 200
    #histos = []
    y_names = [["Water Use Efficiency", "wue"],
               ["Evaporative Stress Index", "esi"],
               ["Above Ground Biomass", "agb"]]
    if make_plts:
        fig, axes = plt.subplots(1, len(y_names), figsize=(12, 4))
    for i in range(len(y_predicted)):
        if mode == "geo":
            y_actual[i] = y_actual[i].flatten()
            y_predicted[i] = y_predicted[i].flatten()
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_actual[i], y_predicted[i])
        r_values.append(r_value)
        if make_plts:
            histo_i = np.histogram2d(y_actual[i], y_predicted[i], bins=nbins, 
                                     range=[[0,1], [0,1]])[0].transpose()
            histo_i = np.log10(histo_i)
            binned_i = axes[i].imshow(histo_i, origin='lower', extent=[0,1,0,1])
            x_bf = np.linspace(0, 1, 100)
            y_bf = (x_bf * slope) + intercept
            bestfit_i = axes[i].plot(x_bf, y_bf, label='$R^{2}=$' + str(round(r_value, 3)), 
                                     color="red", alpha=0.2)
            axes[i].legend(loc='lower right')
            axes[i].set(xlabel="True Value", title=y_names[i][0])
            axes[i].set(ylabel="Predicted Value")
            fig.colorbar(binned_i, ax=axes[i], aspect=20, shrink=0.75)
    if make_plts:
        axes[1].set(title="Predicted Values Versus True Values: " + model_name + "\n" + y_names[1][0])
        plt.savefig("../visualize/y_yhat_train/" + model_name + "_log.png")
    return r_values

def compute_metrics(working_model, val_wrangler, metrics_params, make_plts=False):
    metric_layer = {}
    y_actual, y_predicted = working_model.predict(val_wrangler)
    metric_layer["mse"] = metric_mse(y_predicted, y_actual, "flattened", metrics_params)
    metric_layer["mae"] = metric_mae(y_predicted, y_actual, "flattened", metrics_params)
    metric_layer["r2"] = metric_r2(y_predicted, y_actual, "geo", make_plts, working_model.name)
    return metric_layer


def reformat_metrics(in_metrics, metrics_params, y_layers):
    ### come in as [fold, metric_type, granularity, ylayer]
    ### prefer... [metric_type+granularity, ylayer, fold]
    ref_metrics = {}
    ### copy over mse, mae
    for mtype in ["mse", "mae"]:
        for k in range(len(metrics_params)):
            gtype = metrics_params[k]
            ref_metrics[mtype + "_" + gtype] = []
            for i in range(len(y_layers)):
                ref_metrics[mtype + "_" + gtype].append([])
                for j in range(len(in_metrics)):
                    ref_metrics[mtype + "_" + gtype][i].append(in_metrics[j][mtype][k][i])
    ref_metrics["r2"] = []
    for i in range(len(y_layers)):
        ref_metrics["r2"].append([])
        for j in range(len(in_metrics)):
            ref_metrics["r2"][i].append(in_metrics[j]["r2"][i])

    return ref_metrics

def save_metrics(computed_metrics, wmdir, model_mode):
    with open(wmdir + "/metrics" + model_mode + ".txt", "wb") as metric_out:
        pickle.dump(computed_metrics, metric_out)

def process_dims(layer_dims, x_ids, y_ids):
    ### compute combinations
    unique_layer_dims = []
    unique_ydims = []

    for i in range(len(layer_dims)):
        if i < len(x_ids) and layer_dims[i] not in unique_layer_dims:
            unique_layer_dims.append(layer_dims[i])
        elif i >= len(x_ids) and layer_dims[i] not in unique_ydims:
            unique_ydims.append(layer_dims[i])
    sort_unique = list(unique_layer_dims)
    sort_unique.sort()
    unique_freq = [0 for ii in range(len(sort_unique))]
    unique_ydims.sort()
    unique_yfreq = [0 for ii in range(len(unique_ydims))]
    for i in range(len(layer_dims)):
        if i < len(x_ids):
            for j in range(len(sort_unique)):
                if layer_dims[i] == sort_unique[j]:
                    unique_freq[j] += 1
        else:
            for j in range(len(unique_ydims)):
                if layer_dims[i] == unique_ydims[j]:
                    unique_yfreq[j] += 1

    return sort_unique, unique_ydims, unique_freq, unique_yfreq
