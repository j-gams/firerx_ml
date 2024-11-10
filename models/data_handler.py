### data obj

from osgeo import gdal
import numpy as np
import h5py
import keras.utils as kr_utils
import matplotlib.pyplot as plt

class data_wrangler (kr_utils.Sequence):
    def __init__ (self, rootdir, n_layers, on_folds, cube_dims, batch_size, buffer_nodata, x_ids, y_ids, sample_weights, low_mem=True, **kwargs):
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
        self.on_folds = on_folds
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

        ### load index data
        self.test_index = np.genfromtxt(rootdir + "/test.csv", delimiter=',')
        self.combined_index = np.genfromtxt(rootdir + "/remaining.csv", delimiter=',')
        self.src_test = np.array(self.test_index)
        self.src_comb = np.array(self.combined_index)
        self.train_ids = []
        self.src_trn = []
        self.val_ids = []
        self.src_val = []
        for f in on_folds:
            self.train_ids.append(np.genfromtxt(rootdir+"/train_"+str(f)+".csv", delimiter=','))
            self.val_ids.append(np.genfromtxt(rootdir+"/val_"+str(f)+".csv", delimiter=','))
            self.src_trn.append(self.train_ids[-1])
            self.src_val.append(self.val_ids[-1])

        ### load normalization data
        self.combined_min = np.genfromtxt(rootdir + "/norm_layer_mins_combined.csv", delimiter=',')
        self.combined_max = np.genfromtxt(rootdir + "/norm_layer_maxs_combined.csv", delimiter=',')
        self.fold_mins = []
        self.fold_maxs = []
        for f in self.on_folds:
            self.fold_mins.append(np.genfromtxt(rootdir + "/norm_layer_mins_fold_"+str(f)+".csv", delimiter=','))
            self.fold_maxs.append(np.genfromtxt(rootdir + "/norm_layer_maxs_fold_"+str(f)+".csv", delimiter=','))

        self.setup("train", loading=False)

    def setup(self, mode, loading=True):
        ### set mode and deal with indices...
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

        ### if not low_memory mode load all the layers into memory 

        if loading and not self.low_memory:
            for i in range(self.n_layers):
                self.h5_data[i] = np.array(self.h5_data[i])

    def iter_fold(self):
        self.train_fold = (self.train_fold + 1) % len(self.on_folds)

    def set_fold(self, fold):
        if self.mode == "train":
            self.train_fold = fold
        else:
            self.train_fold = 0

    def exclude_ids(self, exclude):
        self.cache_x_ids = list(self.x_ids)
        self.x_ids = [x_id for x_id in self.x_ids if x_id not in exclude]
        print("before exclusion", self.cache_x_ids)
        print("remaining after exclusion", self.x_ids)

    def include_all(self):
        self.x_ids = self.cache_x_ids

    def get_h5_data(self, layerid):
        print("get h5 ", layerid)
        if self.mode == "train":
            ### we do not have the data loaded
            ### need to get these wrt underlying indexing not shuffled indexing
            if self.low_memory:
                return np.array(self.h5_src[layerid]["data"])[self.src_trn[self.train_fold]]
            ### we have the data loaded already
            #print("issueshape", self.h5_data[layerid].shape, type(self.train_ids[self.train_fold]))
            return self.h5_data[layerid][self.src_trn[self.train_fold].astype(int)]
        elif self.mode == "combine":
            ### we do not have the data loaded
            if self.low_memory:
                return np.array(self.h5_src[layerid]["data"])[self.src_comb]
            ### we have the data loaded already
            return self.h5_data[layerid][self.src_comb_.astype(int)]

    def set_sample_weights(self, sample_weights):
        print("confirming set sample weights")
        if self.use_sample_weights:
            self.sample_weights = sample_weights
            ### need to iterate through and move array to reflect indexing
            ### shape of [fold][y][sample]
            for j in range(len(sample_weights)):
                for i in range(len(self.use_y_ids)):
                    t_sw = np.ones(len(self.h5_data[self.use_y_ids[i]]))
                    if self.mode == "train":
                        t_sw[self.src_trn[self.train_fold].astype(int)] = sample_weights[j][i]
                    elif self.mode == "val":
                        t_sw[self.src_trn[self.train_fold].astype(int)] = sample_weights[j][i]
                    elif self.mode == "combine":
                        t_sw[self.src_comb[self.train_fold].astype(int)] = sample_weights[j][i]
                    elif self.mode == "test":
                        t_sw[self.src_testl[self.train_fold].astype(int)] = sample_weights[j][i]
                    self.sample_weights[j][i] = t_sw

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
        
    def getsw(self, idx):
        ret_sw = []
        for j in range(len(self.use_y_ids)):
            if self.mode == "train":
                ret_sw.append(self.sample_weights[self.train_fold][j][idx].reshape((len(idx), -1)))
            elif self.mode == "val":
                ret_sw.append(self.sample_weights[self.train_fold][j][idx].reshape((len(idx), -1)))
            elif self.mode == "combine":
                ret_sw.append(self.sample_weights[0][j][idx].reshape((len(idx), -1)))
            elif self.mode == "test":
                ret_sw.append(self.sample_weights[0][j][idx].reshape((len(idx), -1)))

        return ret_sw

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
            #ret_sw = [self.sample_weights[self.train_fold][jj][ret_indices] for jj in range(len(self.use_y_ids))]
            ret_sw = self.getsw(ret_indices)
            
            return tuple(ret_x), tuple(ret_y), tuple(ret_sw)
        return tuple(ret_x), tuple(ret_y)

    def on_epoch_end(self):
        self.shuffle()