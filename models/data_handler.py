### data obj

from osgeo import gdal
import numpy as np
import h5py
import keras.utils as kr_utils
import matplotlib.pyplot as plt

class data_wrangler (kr_utils.Sequence):
    def __init__ (self, rootdir, n_layers, n_folds, cube_dims, batch_size, buffer_nodata, x_ids, y_ids, sample_weights, low_mem=True, **kwargs):
        ### handle basic parameters
        self.n_layers = n_layers
        self.buffer_nodata = buffer_nodata
        self.cube_res = cube_dims
        self.x_ids = x_ids
        self.y_ids = y_ids
        self.use_y_ids = list(y_ids)
        self.batch_size = batch_size
        ### possible modes -- {train, val, combine, test}
        self.mode = "train"
        self.train_fold = 0
        self.n_folds = n_folds
        self.use_sample_weights = sample_weights
        self.sample_weights = []
        self.low_memory = low_mem

        ### prepare for h5 data
        self.layer_locs = []
        self.h5_src = []
        self.h5_data = []
        ### setup references for h5 data
        for i in range(n_layers):
            self.layer_locs.append(rootdir + "/layer_"+str(i)+".h5")
            self.h5_src.append(h5py.File(self.layer_locs[i], 'r'))
            self.h5_data.append(self.h5_src[i]["data"])
        ### if not low_memory mode load all the layers into memory 
        if not self.low_memory:
            for i in range(n_layers):
                self.h5_data[i] = np.array(self.h5_data[i])

        ### load index data
        self.test_index = np.genfromtxt(rootdir + "/test.csv", delimiter=',')
        self.combined_index = np.genfromtxt(rootdir + "/remaining.csv", delimiter=',')
        self.train_ids = []
        self.val_ids = []
        for i in range(n_folds):
            self.train_ids.append(np.genfromtxt(rootdir+"/train_"+str(i)+".csv", delimiter=','))
            self.val_ids.append(np.genfromtxt(rootdir+"/val_"+str(i)+".csv", delimiter=','))

        ### load normalization data
        self.combined_min = np.genfromtxt(rootdir + "/norm_layer_mins_combined.csv", delimiter=',')
        self.combined_max = np.genfromtxt(rootdir + "/norm_layer_maxs_combined.csv", delimiter=',')
        self.fold_mins = []
        self.fold_maxs = []
        for i in range(self.n_folds):
            self.fold_mins.append(np.genfromtxt(rootdir + "/norm_layer_mins_fold_"+str(i)+".csv", delimiter=','))
            self.fold_maxs.append(np.genfromtxt(rootdir + "/norm_layer_maxs_fold_"+str(i)+".csv", delimiter=','))

        self.set_mode("train")

    def set_mode(self, mode):
        self.mode = mode
        self.index_len = 0
        if self.mode == "train":
            self.index_len = self.train_ids[self.train_fold].shape[0]
        elif self.mode == "val":
            self.index_len = self.val_ids[self.train_fold].shape[0]
        elif self.mode == "combine":
            self.index_len = self.combined_index.shape[0]
        elif self.mode == "test":
            self.index_len = self.test_index.shape[0]
        self.lenn = int(np.ceil(self.index_len / self.batch_size))
        self.shuffle()

    def set_fold(self, fold):
        self.train_fold = fold

    def exclude_ids(self, exclude):
        self.cache_x_ids = list(self.x_ids)
        self.x_ids = [x_id for x_id in self.x_ids if x_id not in exclude]
        print("before exclusion", self.cache_x_ids)
        print("remaining after exclusion", self.x_ids)

    def include_all(self):
        self.x_ids = self.cache_x_ids

    def get_h5_data(self, layerid):
        ### we do not have the data loaded
        if self.low_memory:
            return np.array(self.h5_src[layerid]["data"])
        ### we have the data loaded already
        else:
            return self.h5_data[layerid]

    """def load_sample_weights(self, loc, nbins):
        pass

    def compute_sample_weight(self, mode, nbins, vis=False):
        self.sample_weights = [[] for ii in range(len(self.use_y_ids))]
        valuefreqs = []
        target_names = ["WUE", "ESI", "AGB"]
        colors = ["salmon", "lightgreen", "lightblue"]
        if mode == "real_bininv":
            print("computing sample weights (real_bininv)...", end="", flush=True)
            ### iterate over y layers and for each, bin and come up with thresholds
            for i in range(len(self.use_y_ids)):
                valuefreq = np.log(np.histogram(self.apply_norm(self.h5_data[self.use_y_ids[i]], self.use_y_ids[i]), bins=nbins, range=[0, 1])[0])
                #print(valuefreq.shape)
                if vis:
                    plt.bar(np.arange(nbins)/nbins, valuefreq, width=1/nbins, color=colors[i])
                    plt.title(target_names[i] + " Log Normalized Pixel Value Frequency (" + str(nbins) + " bins)")
                    plt.xlabel("Normalized Value")
                    plt.ylabel("Log Bin Frequency")
                    plt.savefig("../visualize/data_dist/sample_freq_plot_" + str(i) + ".png")
                    plt.clf()
                for j in range(nbins):
                    if valuefreq[j] == 0:
                        valuefreq[j] = 1
                    else:
                        valuefreq[j] = (self.h5_data[self.use_y_ids[i]].shape[1]**2)/valuefreq[j]
                temp_max = np.max(valuefreq)
                for j in range(nbins):
                    valuefreq[j] /= temp_max
                valuefreqs.append(np.array(valuefreq))
                plt.bar(np.arange(nbins)/nbins, valuefreqs[i], width=1/nbins, color=colors[i])
                plt.title(target_names[i] + " Pixel Value Bin Weights (" + str(nbins) + " bins)")
                plt.xlabel("Normalized Value (" + str(nbins) + " bins)")
                plt.ylabel("Pixel Value Bin Weight")
                plt.savefig("../visualize/data_dist/sample_weight_plot_" + str(i) + ".png")
                plt.clf()
            ### values for each class of sample... now map samples to values
            print("halfway...", end="", flush=True)
            for i in range(len(self.use_y_ids)):
                for k in range(len(self.h5_data[self.use_y_ids[i]])):
                    ### idea 1... sum pixel weights over sample
                    ### idea 2... know nbins and range... so multiply by nbins, convert to int to get bin number...?
                    ### get bins, then valuefreqs at locations
                    self.sample_weights[i].append(np.sum(valuefreqs[i][np.clip((self.apply_norm(self.h5_data[self.use_y_ids[i]][k], self.use_y_ids[i]) * nbins), None, nbins-1).astype(int)]))
                    #self.sample_weights[i].append(valuefreqs[i][np.clip((self.apply_norm(self.h5_data[self.use_y_ids[i]][k], self.use_y_ids[i]) * nbins), None, nbins-1).astype(int)]) 
            print("done")
            for i in range(len(self.use_y_ids)):
                self.sample_weights[i] = np.array(self.sample_weights[i])
                print(self.sample_weights[i].shape)
        if vis:
            plt.close()"""

    def set_sample_weights(self, sample_weights):
        print("confirming set sample weights")
        self.sample_weights = sample_weights

    def shuffle(self):
        if self.mode == "train":
            np.random.shuffle(self.train_ids[self.train_fold])
        elif self.mode == "val":
            np.random.shuffle(self.val_ids[self.train_fold])
        elif self.mode == "combine":
            np.random.shuffle(self.combined_index)
        elif self.mode == "test":
            np.random.shuffle(self.test_index)


    def getindices(self, idx):
        if self.mode == "train":
            return self.train_ids[self.train_fold][idx*self.batch_size: min(((idx+1) * self.batch_size), self.index_len)]
        elif self.mode == "val":
            return self.val_ids[self.train_fold][idx*self.batch_size: min(((idx+1) * self.batch_size), self.index_len)]
        elif self.mode == "combine":
            return self.combined_index[idx*self.batch_size: min(((idx+1) * self.batch_size), self.index_len)]
        elif self.mode == "test":
            return self.test_index[idx*self.batch_size: min(((idx+1) * self.batch_size), self.index_len)]

    def __len__ (self):
        return self.lenn

    def apply_norm(self, npar, k):
        if self.mode == "train" or self.mode == "val":
            return (npar - self.fold_mins[self.train_fold][k]) / (self.fold_maxs[self.train_fold][k] - 
                                                               self.fold_mins[self.train_fold][k])
        else:
            return (npar - self.combined_min[k]) / (self.combined_max[k] - self.combined_min[k])

    def set_single_y(self, set_to):
        self.use_y_ids = [self.y_ids[set_to]]
    def set_multi_y(self):
        self.use_y_ids = list(self.y_ids)

    def __getitem__ (self, idx):
        ### load cubes
        ### apply normalization
        ### return
        ret_indices = np.sort(self.getindices(idx)).astype(int)
        ### ret x is formatted [layer][batch, i, j]
        
        ret_x = []
        ret_y = []
        for i in range(len(self.x_ids)):
            ret_x.append(np.zeros((len(ret_indices), self.cube_res[self.x_ids[i]], self.cube_res[self.x_ids[i]])))
        for i in range(len(self.use_y_ids)):
            ret_y.append(np.zeros((len(ret_indices), self.cube_res[self.use_y_ids[i]] * self.cube_res[self.use_y_ids[i]])))
        for j in range(len(self.x_ids)):
            ret_x[j][:,:] = self.apply_norm(np.array(self.h5_data[self.x_ids[j]][ret_indices, :, :]), self.x_ids[j])
        for j in range(len(self.use_y_ids)):
            ret_y[j][:,:] = self.apply_norm(np.array(self.h5_data[self.use_y_ids[j]][ret_indices, :, :]), self.use_y_ids[j]).reshape((len(ret_indices), -1))
            #ret_y.append(self.apply_norm(np.array(self.h5_data[self.use_y_ids[j]][ret_indices, :, :]), self.use_y_ids[j]).flatten())
        if self.use_sample_weights:
            ret_sw = []
            for j in range(len(self.use_y_ids)):
                ret_sw.append(self.sample_weights[j][ret_indices].reshape((len(ret_indices), -1)))
            #print(self.mode, ret_y[0].shape, ret_sw[0].shape, "*", ret_y[1].shape, ret_sw[1].shape, "*", ret_y[2].shape, ret_sw[2].shape)
            return tuple(ret_x), tuple(ret_y), tuple(ret_sw)
        return tuple(ret_x), tuple(ret_y)

    def on_epoch_end(self):
        self.shuffle()