import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate, Flatten, Reshape, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Layer, Embedding, LayerNormalization, MultiHeadAttention, Add, Dropout

from tensorflow import shape as tfshape
from tensorflow import image as tfimage
from tensorflow import reshape as tfreshape
from tensorflow import range as tfrange
from tensorflow import concat as tfconcat
from tensorflow import constant as tfconstant
from tensorflow import map_fn as tfmap_fn
from tensorflow import gather_nd as tfgather_nd
from tensorflow.nn import gelu

from data_handler import data_wrangler
import math
import pickle

import time

### Contents...
### - model maker
### - losscallback
### - Cascade 1
### - Flat 1

### model importer
def make_model(modeltype):
    if "cascade1" in modeltype:
        return model_cascade1()
    elif "cascade2" in modeltype:
        return model_cascade2()
    elif "flat1" in modeltype:
        return model_flat1()
    elif "flat2" in modeltype:
        return model_flat2()
    elif "dummy" in modeltype:
        return model_dummy()
    elif "transformer1" in modeltype:
        return multi_vit()

def save_metrics(metrics_dict, modeldir, fold):
    #self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
    with open(modeldir + "/metric_" + str(fold) + ".txt", "wb") as metric_out:
        pickle.dump(metrics_dict, metric_out)

class lossCallback(Callback):
    def __init__ (self):
        self.logs = dict()
        self.init = False
        self.lasttime = -1

    def on_epoch_end(self, epoch, logs={}):
        keys = list(logs.keys())
        if not self.init:
            for k in keys:
                self.logs[k] = []
            self.init = True
            self.logs["time"] = []
        #['loss', 'accuracy', 'val_loss', 'val_accuracy']
        #print("CALLBACK: epoch", epoch, "keys:", keys)
        ctime = time.time()
        self.logs["time"].append(ctime)
        self.lasttime = ctime
        for k in keys:
            self.logs[k].append(logs.get(k))

    def resume_from_load(self, loaded_logs):
        self.init = True
        self.logs = loaded_logs

### DUMMY MODEL
class model_dummy:
    def __init__(self):
        self.base_model = None

    def make_base_model(self):
        print("setting up dummy model :)")
        return None

    def setup(self, model_params, model_name):
        self.make_base_model()

    def load(self):
        pass

    def fit(self, train_data, val_data, n_epochs):
        print("fitting dummy model")


    def predict(self, val_data):
        print("predicting with dummy model")
        return None

    def save(self):
        pass

### MODEL: Cascade 1
class model_cascade1:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq):
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]
        inputs = []
        convs1 = [[] for ii in range(len(uniquedimssorted))]
        for i in range(len(xdims)):
            inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
            if xdims[i] == uniquedimssorted[0]:
                ### 1 -> 1
                c1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1))(inputs[i])
                convs1[0].append(c1)
            elif xdims[i] == uniquedimssorted[1]:
                ### 2 -> 2
                c1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1))(inputs[i])
                convs1[1].append(c1)
            elif xdims[i] == uniquedimssorted[2]:
                ### 34 -> 17
                c1 = Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="same")(inputs[i])
                convs1[2].append(c1)

        ###

        convs2 = [[] for ii in range(len(uniquedimssorted) - 1)]
        for i in range(len(convs1)):
            if i == 0:
                ### 1 -> 1
                convs2[0].append(convs1[0][0])
            else:
                c2 = Concatenate(axis=3)(convs1[i])
                if i == 1:
                    ### 2 -> 2
                    c2 = Conv2D(filters=8 * freq[1], kernel_size=(1, 1), strides=(1, 1))(c2)
                if i == 2:
                    ### 17 -> 8
                    c2 = Conv2D(filters=8 * 2 * freq[2], kernel_size=(3, 3), strides=(2, 2), padding="valid")(c2)
                    ### 8 -> 2
                    c2 = Conv2D(filters=8 * 4 * freq[2], kernel_size=(5, 5), strides=(4, 4), padding="same")(c2)
                convs2[1].append(c2)

        ###

        convs3 = []
        for i in range(len(convs2)):
            if i == 0:
                convs3.append(convs2[0][0])
            else:
                c3 = Concatenate(axis=3)(convs2[i])
                ### 2 -> 2
                c3 = Conv2D(filters=8 * 4 * (freq[1] + freq[2]), kernel_size=(1, 1), strides=(1, 1))(c3)
                ### 2 -> 1
                c3 = Conv2D(filters=8 * 8 * (freq[1] + freq[2]), kernel_size=(2, 2), strides=(2, 2))(c3)
                convs3.append(c3)

        ### 1x1x(64*18)
        c4 = Concatenate(axis=3)(convs3)
        c4 = Conv2D(filters=8 * 8 * (freq[1] + freq[2] + freq[0]), kernel_size=(1, 1), strides=(1, 1))(c4)
        c5 = Flatten()(c4)

        c5 = Dense(1600, activation="relu")(c5)
        c5 = Dense(1200, activation="relu")(c5)
        c5 = Dense(1600, activation="relu")(c5)
        ysplit = []
        for j in range(len(ydims_unique)):
            yj = Dense((ydims_unique[j] ** 2) * ydims_freq[j], activation="relu")(c5)
            ysplit.append(yj)
        associated_dim = []
        ysplit2 = []
        yfinalnames = ["agb", "wue", "esi"]
        ncounter = 0
        for j in range(len(ydims_unique)):
            if ydims_freq[j] == 1:
                yj2 = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(ysplit[j])
                ysplit2.append(yj2)
                associated_dim.append(ydims_unique[j])
                ncounter += 1
            else:
                for i in range(ydims_freq[j]):
                    yj2a = Dense(ydims[j] ** 2, activation="relu")(ysplit[j])
                    yj2b = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(yj2a)
                    ysplit2.append(yj2b)
                    associated_dim.append(ydims_unique[j])
                    ncounter += 1
        yfinal = []
        for i in range(len(ydims)):
            ### find one w/ appropriate exit dim
            for j in range(len(ysplit2)):
                if associated_dim[j] == ydims[i]:
                    yfinal.append(ysplit2.pop(j))
                    associated_dim.pop(j)
                    break

        return Model(inputs=inputs, outputs=yfinal)

    def setup(self, model_params, model_name):
        self.v = model_params["verbosity"]
        self.layer_dims = model_params["layerdims"]
        self.name = model_name
        self.modeldir = model_params["dir"]
        self.x_ids = model_params["x_layers"]
        self.y_ids = model_params["y_layers"]

        self.training_loss = model_params["training_loss"]
        self.monitor_loss = model_params["monitor_loss"]
        ### compute combinations
        unique_layer_dims = []
        unique_ydims = []
        # layer_cat = []

        print(self.x_ids, self.y_ids)
        ### TODO - make this not dumb
        # layer_cat = []
        for i in range(len(self.layer_dims)):
            if i in self.x_ids and self.layer_dims[i] not in unique_layer_dims:
                unique_layer_dims.append(self.layer_dims[i])
            if i in self.y_ids and self.layer_dims[i] not in unique_ydims:
                unique_ydims.append(self.layer_dims[i])

        # for i in range(len(unique_layer_dims)):
        #    for j in range(len(layer_dims)):
        #        if layer_dims[j] == unique_layer_dims[i]:
        #            layer_cat.append(i)

        sort_unique = list(unique_layer_dims)
        sort_unique.sort()
        unique_freq = [0 for ii in range(len(sort_unique))]
        unique_ydims.sort()
        unique_yfreq = [0 for ii in range(len(unique_ydims))]
        for i in range(len(self.layer_dims)):
            if i in self.x_ids:
                for j in range(len(sort_unique)):
                    if self.layer_dims[i] == sort_unique[j]:
                        unique_freq[j] += 1
            else:
                for j in range(len(unique_ydims)):
                    if self.layer_dims[i] == unique_ydims[j]:
                        unique_yfreq[j] += 1

        print(unique_layer_dims, unique_freq)
        print(unique_ydims, unique_yfreq)

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], sort_unique,
                                               unique_freq, unique_ydims, unique_yfreq)
        self.base_model.compile(loss=self.training_loss)
        if self.v > 0:
            print(self.base_model.summary())
        callback1 = lossCallback()
        callback2 = ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
                                    monitor = "val_mean_squared_error", verbose=2, mode="min",
                                    save_best_only=True, save_freq="epoch", save_weights_only=True)
        self.callbacks = [callback1]#, callback2]

    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

    def fit(self, train_data, val_data, n_epochs):
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=self.v)

    def predict(self, val_data):
        ### iterate over y layers
        yhats = [[] for iii in range(len(self.y_ids))]
        ys = [[] for iii in range(len(self.y_ids))]
        for i in range(len(val_data)):
            ### val y at batch i
            ys_i = val_data[i][1]
            y_hats_i = self.base_model(val_data[i][0])
            for j in range(len(self.y_ids)):
                for elty in ys_i[j]:
                    ys[j].append(elty)
                for elth in y_hats_i[j]:
                    yhats[j].append(elth)

        for i in range(len(self.y_ids)):
            yhats[i] = np.array(yhats[i])[:, :, :, 0]
            ys[i] = np.array(ys[i])
            print("shape sanity check", i, "-", yhats[i].shape, ys[i].shape)

        self.recent_ys = ys
        self.recent_yhats = yhats
        return ys, yhats

    def save(self):
        self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "wb") as cblog:
            pickle.dump(self.callbacks[0].logs, cblog)

class TensorAddLayer(keras.layers.Layer):
    def __init__(self, adims, bdims):
        super().__init__()
        self.factor = math.ceil(adims / bdims)
        self.adims = adims
        self.bdims = bdims

    def call(self, inputs):
        tiled_tensor = keras.backend.tile(inputs[1], [1, self.factor, self.factor, 1])
        return tiled_tensor + inputs[0]

### MODEL: Cascade 2 - new basic cascade
class model_cascade2:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq, variant="a",
                        singletask=False):
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]
        inputs = []
        convs1 = [[] for ii in range(len(uniquedimssorted))]
        convs2 = []
        if variant == "e":
            evariant_tofreq = max(list(freq))

        print("building variant", variant)
        print(uniquedimssorted)
        print(freq)

        convdims = list(uniquedimssorted)
        sfreq = list(freq)
        ### step 1 - gather layers into groups
        for i in range(len(xdims)):
            inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
            for j in range(len(uniquedimssorted)):
                if xdims[i] == uniquedimssorted[j]:
                    convs1[j].append(inputs[i])
                    break

        ### step 2 - concatenate
        for j in range(len(uniquedimssorted)):
            convs2.append(Concatenate(axis=3)(convs1[j]))

        ### step 3 - action
        for j in range(len(uniquedimssorted)):
            if variant == "a":
                if uniquedimssorted[j] <= 2:
                    ### 1 -> 1 or 2 -> 2
                    convs2[j] = Conv2D(filters=8*freq[j], kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=8*freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "b":
                if uniquedimssorted[j] == 1:
                    ### 1 -> 2
                    convs2[j] = Conv2DTranspose(filters=8*freq[j], kernel_size=(2, 2),
                                                strides=(1, 1))(convs2[j])
                    convdims[j] = 2
                elif uniquedimssorted[j] == 2:
                    ### 2-> 2
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "c":
                if uniquedimssorted[j] <= 2:
                    ### 1 -> 1 or 2 -> 2
                    convs2[j] = Conv2D(filters=8*freq[j], kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=8*freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "d":
                if uniquedimssorted[j] <= 2:
                    ### 1 -> 1 or 2 -> 2
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "e":
                #if uniquedimssorted[j] == 1:
                #    ### 1 -> 1 or 2 -> 2
                #    convs2[j] = Conv2D(filters=2 * evariant_tofreq, kernel_size=(1, 1), strides=(1, 1),
                #                       padding="same")(convs2[j])
                if uniquedimssorted[j] <= 2:
                    ### 2 -> 1
                    convs2[j] = Conv2D(filters=2 * evariant_tofreq, kernel_size=(1, 1), strides=(1, 1),
                                       padding="same")(convs2[j])
                    #convdims[j] = 1
                else:
                    ### 1st conv
                    ### 34 -> 32
                    ### or 27 -> 25
                    convs2[j] = Conv2D(filters=2 * evariant_tofreq, kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2

        if variant == "e":
            ### do special layer...
            evariant_workdim = max(convdims)
            etemp = [convs2[2]]
            if len(sfreq) > 1:
                ## adim bdim [inputa, inputb]
                etemp.append(TensorAddLayer(32, 1)([convs2[2], convs2[0]]))
                etemp.append(TensorAddLayer(32, 2)([convs2[2], convs2[1]]))
            else:
                print("uhoh")
            sfreq = [sum(sfreq)]
            convdims = [convdims[2]]
            convs2[0] = Concatenate(axis=3)(etemp)

        ### step 4 - action:
        for j in range(len(sfreq)):
            if variant == "a":
                if convdims[j] <= 2:
                    pass
                else:
                    ### 1st maxpool
                    ### 32 -> 16
                    ### or 25 -> 13
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j]/2)
            elif variant == "b":
                if convdims[j] <= 2:
                    #convs2[j] = Concatenate(axis=3)(convs2[j])
                    pass
                else:
                    ### 1st maxpool
                    ### 32 -> 16
                    ### or 25 -> 13
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "c":
                if convdims[j] <= 2:
                    #convs2[j] = Concatenate(axis=3)(convs2[j])
                    pass
                else:
                    ### 1st maxpool
                    ### 32 -> 16
                    ### or 25 -> 13
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "d":
                if convdims[j] > 2:
                    ### 1st maxpool
                    ### 32 -> 16
                    ### or 25 -> 13
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "e":
                ## 32 -> 32 or 27 -> 27
                convs2[j] = Conv2D(filters=8 * sfreq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="same")(convs2[j])

        ### step 5 - action:
        for j in range(len(sfreq)):
            if variant == "a":
                if convdims[j] <= 2:
                    pass
                else:
                    ### 2nd conv
                    ### 16 -> 8 and 13 -> 7
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(2, 2),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "b":
                if convdims[j] > 2:
                    ### 2nd conv
                    ### 16 -> 8 and 13 -> 7
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(2, 2),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "c":
                if convdims[j] > 2:
                    ### 1st conv
                    ### 16 -> 14
                    ### or 13 -> 11
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "d":
                if convdims[j] > 2:
                    ### 2nd conv
                    ### 16 -> 6 and 13 -> 5
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(3, 3),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 3)
            elif variant == "e":
                ### 1st maxpool
                ### 32 -> 16
                ### or 25 -> 13
                convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                convdims[j] = math.ceil(convdims[j] / 2)

        ### step 6 - action:
        for j in range(len(sfreq)):
            if variant == "a":
                if convdims[j] <= 2:
                    pass
                else:
                    ### 2nd maxpool
                    ### 8 -> 4 and 7 -> 4
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "b":
                if convdims[j] > 2:
                    ### 2nd maxpool
                    ### 8 -> 4 and 7 -> 4
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "c":
                if convdims[j] > 2:
                    ### 2nd maxpool
                    ### 14 -> 7 and 11 -> 6
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "d":
                if convdims[j] > 2:
                    ### 2nd conv
                    ### 6 -> 2 and 5 -> 2
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(3, 3),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 3)
            elif variant == "e":
                ### 2nd conv
                ### 16 -> 6 and 13 -> 5
                convs2[j] = Conv2D(filters=16 * sfreq[j], kernel_size=(3, 3), strides=(3, 3),
                                   padding="same")(convs2[j])
                convdims[j] = math.ceil(convdims[j] / 3)

        ### step 7 - action:
        for j in range(len(sfreq)):
            if variant == "a":
                if convdims[j] <= 2:
                    pass
                else:
                    ### 3rd conv
                    ### 4 -> 2 and 4 -> 2
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(2, 2),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "b":
                if convdims[j] > 2:
                    ### 3rd conv
                    ### 4 -> 2 and 4 -> 2
                    convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(2, 2),
                                       padding="same")(convs2[j])
                    convdims[j] = math.ceil(convdims[j] / 2)
            elif variant == "c":
                ### 3rd conv
                if convdims[j] > 2:
                    ### 7 -> 5
                    ### or 6 -> 4
                    convs2[j] = Conv2D(filters=8 * freq[j], kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid")(convs2[j])
                    convdims[j] -= 2
            elif variant == "e":
                ### 2nd conv
                ### 6 -> 2 and 5 -> 2
                convs2[j] = Conv2D(filters=16 * freq[j], kernel_size=(3, 3), strides=(3, 3),
                                   padding="same")(convs2[j])
                convdims[j] = math.ceil(convdims[j] / 3)

        ### step 7.5 - action:
        for j in range(len(sfreq)):
            if variant == "c":
                if convdims[j] > 2:
                    ### 3rd maxpool
                    ### 5 ->2 and 4->2
                    convs2[j] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(convs2[j])
                    convdims[j] = 2

        print(convdims)
        run_concat = False
        if variant == "a":
            if len(convdims) > 1:
                convdims = [1, 2]
                convs2[1] = [convs2[1], convs2[2]]
                sfreq[1] = freq[1] + freq[2]
                run_concat = True
        if variant == "c":
            if len(convdims) > 1:
                convdims = [1, 2]
                convs2[1] = [convs2[1], convs2[2]]
                sfreq[1] = freq[1] + freq[2]
                run_concat = True
        if variant == "d":
            if len(convdims) > 1:
                convdims = [1, 2]
                convs2[1] = [convs2[1], convs2[2]]
                sfreq[1] = freq[1] + freq[2]
                run_concat = True
        ### e is already concatted

        ### step 8 - concat:
        for j in range(len(convdims)):
            if variant == "a" or variant == "c" or variant == "d":
                if convdims[j] == 2 and run_concat:
                    convs2[j] = Concatenate(axis=3)(convs2[j])
                else:
                    print(j, convdims[j])

        ### step 9 - conv
        for j in range(len(convdims)):
            if variant == "a":
                if convdims[j] == 2:
                    convs2[j] = Conv2D(filters=16 * sfreq[j], kernel_size=(2, 2), strides=(2, 2),
                           padding="same")(convs2[j])
            if variant == "c":
                if convdims[j] == 2:
                    convs2[j] = Conv2D(filters=16 * sfreq[j], kernel_size=(2, 2), strides=(2, 2),
                           padding="same")(convs2[j])
            if variant == "d":
                if convdims[j] == 2:
                    convs2[j] = Conv2D(filters=16 * sfreq[j], kernel_size=(2, 2), strides=(2, 2),
                           padding="same")(convs2[j])
                    convdims[j] = 1

        run_concat = False
        if variant == "a":
            if len(convdims) > 1:
                convdims = [1]
                convs2[0] = [convs2[0], convs2[1]]
                sfreq[0] += sfreq[0]
                run_concat = True
        if variant == "b":
            if len(convdims) > 1:
                convdims = [2, 2, 2]
                convs2[0] = [convs2[0], convs2[1], convs2[2]]
                sfreq[0] = freq[0] + freq[1] + freq[2]
                convs2[0] = Concatenate(axis=3)(convs2[0])
            else:
                print("nodims")
        if variant == "c":
            if len(convdims) > 1:
                convdims = [1]
                convs2[0] = [convs2[0], convs2[1]]
                sfreq[0] += sfreq[1]
                run_concat = True
        if variant == "d":
            if len(convdims) > 1:
                convdims = [1]
                convs2[0] = [convs2[0], convs2[1]]
                sfreq[0] += sfreq[1]
                run_concat = True
            print(convdims)

        ### step 10 - concat:
        for j in range(len(convdims)):
            if variant == "a" or variant == "c" or variant == "d":
                if convdims[j] == 1 and len(convdims) > 1:
                    convs2[j] = Concatenate(axis=3)(convs2[j])
                else:
                    print("oh no", j, convdims[j], len(convs2))

        ### step 12 - conv and flatten
        convs2[0] = Conv2D(filters=16 * sfreq[0], kernel_size=(1, 1), strides=(1, 1),
                           padding="same")(convs2[0])
        convres = Flatten()(convs2[0])
        #convres = Concatenate()(convs2)

        fc = Dense(1600, activation="relu")(convres)
        fc = Dense(1800, activation="relu")(fc)
        fc = Dense(1600, activation="relu")(fc)
        ysplit = []
        if singletask is None:
            for j in range(len(ydims_unique)):
                yj = Dense((ydims_unique[j] ** 2) * ydims_freq[j], activation="relu")(fc)
                ysplit.append(yj)
        else:
            ysplit = Dense((ydims[singletask] ** 2), activation="relu")(fc)
        associated_dim = []
        ysplit2 = []
        yfinalnames = ["agb", "wue", "esi"]
        ncounter = 0
        combineout = True
        if singletask is None:
            if combineout:
                for j in range(len(ydims_unique)):
                    if ydims_freq[j] == 1:
                        yj2 = Dense((ydims_unique[j] ** 2), activation="relu")(ysplit[j])
                        yj2 = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(yj2)
                        ysplit2.append(yj2)
                        associated_dim.append(ydims_unique[j])
                        ncounter += 1
                    else:
                        yj2i = Concatenate()(ysplit)
                        yj2i = Dense(ydims[0] ** 2 + ydims[1] ** 2, activation="relu")(yj2i)
                        for i in range(ydims_freq[j]):
                            yj2a = Dense(ydims[j] ** 2, activation="relu")(yj2i)
                            yj2b = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(yj2a)
                            ysplit2.append(yj2b)
                            associated_dim.append(ydims_unique[j])
                            ncounter += 1
            else:
                for j in range(len(ydims_unique)):
                    if ydims_freq[j] == 1:
                        yj2 = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(ysplit[j])
                        ysplit2.append(yj2)
                        associated_dim.append(ydims_unique[j])
                        ncounter += 1
                    else:
                        for i in range(ydims_freq[j]):
                            yj2a = Dense(ydims[j] ** 2, activation="relu")(ysplit[j])
                            yj2b = Reshape((ydims_unique[j], ydims_unique[j], 1), name=yfinalnames[ncounter])(yj2a)
                            ysplit2.append(yj2b)
                            associated_dim.append(ydims_unique[j])
                            ncounter += 1
        else:
            ysplit2 = Reshape((ydims[singletask], ydims[singletask], 1), name=yfinalnames[singletask])(ysplit)
            associated_dim = ydims[singletask]
            ncounter += 1
        yfinal = []
        if singletask is None:
            for i in range(len(ydims)):
                ### find one w/ appropriate exit dim
                for j in range(len(ysplit2)):
                    if associated_dim[j] == ydims[i]:
                        yfinal.append(ysplit2.pop(j))
                        associated_dim.pop(j)
                        break
        else:
            yfinal = ysplit2
        print("finished setting up")
        return Model(inputs=inputs, outputs=yfinal)

    def setup(self, model_params, model_name):
        self.v = model_params["verbosity"]
        self.layer_dims = model_params["layerdims"]
        self.name = model_name
        self.modeldir = model_params["dir"]
        self.x_ids = model_params["x_layers"]
        self.y_ids = model_params["y_layers"]

        self.training_loss = model_params["training_loss"]
        self.monitor_loss = model_params["monitor_loss"]
        self.vari = model_params["variant"]
        self.singletask = model_params["singletask"]
        ### compute combinations
        unique_layer_dims = []
        unique_ydims = []
        # layer_cat = []

        print(self.x_ids, self.y_ids)
        ### TODO - make this not dumb
        # layer_cat = []
        for i in range(len(self.layer_dims)):
            if i in self.x_ids and self.layer_dims[i] not in unique_layer_dims:
                unique_layer_dims.append(self.layer_dims[i])
            if i in self.y_ids and self.layer_dims[i] not in unique_ydims:
                unique_ydims.append(self.layer_dims[i])

        # for i in range(len(unique_layer_dims)):
        #    for j in range(len(layer_dims)):
        #        if layer_dims[j] == unique_layer_dims[i]:
        #            layer_cat.append(i)

        sort_unique = list(unique_layer_dims)
        sort_unique.sort()
        unique_freq = [0 for ii in range(len(sort_unique))]
        unique_ydims.sort()
        unique_yfreq = [0 for ii in range(len(unique_ydims))]
        for i in range(len(self.layer_dims)):
            if i in self.x_ids:
                for j in range(len(sort_unique)):
                    if self.layer_dims[i] == sort_unique[j]:
                        unique_freq[j] += 1
            else:
                for j in range(len(unique_ydims)):
                    if self.layer_dims[i] == unique_ydims[j]:
                        unique_yfreq[j] += 1

        print(unique_layer_dims, unique_freq)
        print(unique_ydims, unique_yfreq)

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], sort_unique,
                                               unique_freq, unique_ydims, unique_yfreq,
                                               variant=self.vari, singletask=self.singletask)
        print("got here")
        self.base_model.compile(loss=self.training_loss)
        if self.v > 0:
            print(self.base_model.summary())
        callback1 = lossCallback()
        #callback2 = ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
        #                            monitor = "val_mean_squared_error", verbose=2, mode="min",
        #                            save_best_only=True, save_freq="epoch", save_weights_only=True)
        self.callbacks = [callback1]#, callback2]

    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

    def fit(self, train_data, val_data, n_epochs):
        if self.singletask is not None:
            train_data.set_single_y(self.singletask)
            val_data.set_single_y(self.singletask)
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=self.v)
        if self.singletask is not None:
            train_data.set_multi_y()
            val_data.set_multi_y()

    def predict(self, val_data):
        if self.singletask is not None:
            val_data.set_single_y(self.singletask)
        yhats = [[] for iii in range(len(val_data.use_y_ids))]
        ys = [[] for iii in range(len(val_data.use_y_ids))]
        ### iterate over batches?
        for i in range(len(val_data)):
            ### val y at batch i
            ys_i = val_data[i][1]
            y_hats_i = self.base_model(val_data[i][0])
            if self.singletask is not None:
                y_hats_i = [y_hats_i]
                #print(y_hats_i.shape)
            ### iterate over y layers
            for j in range(len(val_data.use_y_ids)):
                for elty in ys_i[j]:
                    ys[j].append(elty)
                for elth in y_hats_i[j]:
                    yhats[j].append(elth)

        for i in range(len(val_data.use_y_ids)):
            yhats[i] = np.array(yhats[i])[:, :, :, 0]
            ys[i] = np.array(ys[i])
            print("shape sanity check", i, "-", yhats[i].shape, ys[i].shape)

        if self.singletask is not None:
            val_data.set_multi_y()

        self.recent_ys = ys
        self.recent_yhats = yhats
        return ys, yhats

    def save(self):
        self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "wb") as cblog:
            pickle.dump(self.callbacks[0].logs, cblog)

### MODEL: Flat 1
class model_flat1:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq):
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]
        inputs = []
        convs = []
        for i in range(len(xdims)):
            inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
            if xdims[i] < 3:
                c1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1))(inputs[i])
                # c1 = Conv2D(filters=16, kernel_size=(1,1), strides=(1,1))(c1)
            else:
                ### 34 -> 32 -> 16 -> 8 -> 4
                c1 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="valid")(inputs[i])
                c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
                c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
                c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
                c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
            convs.append(Flatten()(c1))

        merge = Concatenate()(convs)
        c2 = Dense(4000, activation="relu")(merge)
        c2 = Dense(4000, activation="relu")(c2)
        c2 = Dense(2000, activation="relu")(c2)
        c2 = Dense(1200, activation="relu")(c2)

        y1a = Dense((1 ** 2) * 1, activation="relu")(c2)
        y2a = Dense((15 ** 2) * 2, activation="relu")(c2)

        y1b = Reshape((1, 1, 1), name="agb")(y1a)
        y2b = Dense(15 ** 2, activation="relu")(y2a)
        y2c = Reshape((15, 15, 1), name="wue")(y2b)
        y2d = Dense(15 ** 2, activation="relu")(y2a)
        y2e = Reshape((15, 15, 1), name="esi")(y2d)
        yfinal = [y2c, y2e, y1b]

        return Model(inputs=inputs, outputs=yfinal)

    def setup(self, model_params, model_name):
        self.v = model_params["verbosity"]
        self.layer_dims = model_params["layerdims"]
        self.name = model_name
        self.modeldir = model_params["dir"]
        self.x_ids = model_params["x_layers"]
        self.y_ids = model_params["y_layers"]

        self.training_loss = model_params["training_loss"]
        self.monitor_loss = model_params["monitor_loss"]
        ### compute combinations
        unique_layer_dims = []
        unique_ydims = []
        # layer_cat = []

        ### TODO - make this not dumb
        for i in range(len(self.layer_dims)):
            if i in self.x_ids and self.layer_dims[i] not in unique_layer_dims:
                unique_layer_dims.append(self.layer_dims[i])
            if i in self.y_ids and self.layer_dims[i] not in unique_ydims:
                unique_ydims.append(self.layer_dims[i])
        sort_unique = list(unique_layer_dims)
        sort_unique.sort()
        unique_freq = [0 for ii in range(len(sort_unique))]
        unique_ydims.sort()
        unique_yfreq = [0 for ii in range(len(unique_ydims))]
        for i in range(len(self.layer_dims)):
            if i in self.x_ids:
                for j in range(len(sort_unique)):
                    if self.layer_dims[i] == sort_unique[j]:
                        unique_freq[j] += 1
            else:
                for j in range(len(unique_ydims)):
                    if self.layer_dims[i] == unique_ydims[j]:
                        unique_yfreq[j] += 1

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], unique_layer_dims,
                                               unique_freq, unique_ydims, unique_yfreq)
        self.base_model.compile(loss=self.training_loss)
        if self.v > 0:
            print(self.base_model.summary())
        callback1 = lossCallback()
        callback2 = ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
                                    monitor = "val_mean_squared_error", verbose=2, mode="min",
                                    save_best_only=True, save_freq="epoch", save_weights_only=True)
        self.callbacks = [callback1]#, callback2]
        print("done setting up")

    def load(self):
        print("loading model...")
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

    def fit(self, train_data, val_data, n_epochs):
        print("fitting...")
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=2)

    def predict(self, val_data):
        ### iterate over y layers
        yhats = [[] for iii in range(len(self.y_ids))]
        ys = [[] for iii in range(len(self.y_ids))]
        for i in range(len(val_data)):
            ### val y at batch i
            ys_i = val_data[i][1]
            y_hats_i = self.base_model(val_data[i][0])
            for j in range(len(self.y_ids)):
                for elty in ys_i[j]:
                    ys[j].append(elty)
                for elth in y_hats_i[j]:
                    yhats[j].append(elth)

        for i in range(len(self.y_ids)):
            yhats[i] = np.array(yhats[i])[:,:,:,0]
            ys[i] = np.array(ys[i])
            print("shape sanity check", i, "-", yhats[i].shape, ys[i].shape)
        self.recent_ys = ys
        self.recent_yhats = yhats
        return ys, yhats

    def save(self):
        self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "wb") as cblog:
            pickle.dump(self.callbacks[0].logs, cblog)

### MODEL: Flat 2
class model_flat2:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq):
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]
        inputs = []
        convs = []
        for i in range(len(xdims)):
            """inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
            if xdims[i] < 3:
                c1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1))(inputs[i])
                # c1 = Conv2D(filters=16, kernel_size=(1,1), strides=(1,1))(c1)
            else:
                ### 34 -> 32 -> 16 -> 8 -> 4
                c1 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="valid")(inputs[i])
                c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
                c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
                c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
                c1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
            convs.append(Flatten()(c1))"""
            inputs.append(Input(shape=(xdims[i], xdims[i], 1)))
        c1 = Concatenate(axis=3)(inputs)
        ### 34 -> 32 or 27 -> 25
        c1 = Conv2D(filters=40, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 32 -> 30 or 25 -> 23
        c1 = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 30 -> 28 or 23 -> 21
        c1 = Conv2D(filters=157, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 30 -> 15 or 23 -> 12
        c1 = Conv2D(filters=304, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        ### 15 -> 8 or 12 -> 6
        c1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)

        ### 8 -> 4 or 6 -> 3
        c1 = Conv2D(filters=304, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        ### 4 -> 2 or 3 -> 2
        c1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)
        merge = Flatten()(c1)
        c2 = Dense(1600, activation="relu")(merge)
        c2 = Dense(1800, activation="relu")(c2)
        c2 = Dense(1600, activation="relu")(c2)

        y1a = Dense((1 ** 2) * 1, activation="relu")(c2)
        y2a = Dense((15 ** 2) * 2, activation="relu")(c2)

        y1b = Reshape((1, 1, 1), name="agb")(y1a)
        y2b = Dense(15 ** 2, activation="relu")(y2a)
        y2c = Reshape((15, 15, 1), name="wue")(y2b)
        y2d = Dense(15 ** 2, activation="relu")(y2a)
        y2e = Reshape((15, 15, 1), name="esi")(y2d)
        yfinal = [y2c, y2e, y1b]

        return Model(inputs=inputs, outputs=yfinal)

    def setup(self, model_params, model_name):
        self.v = model_params["verbosity"]
        self.layer_dims = model_params["layerdims"]
        self.name = model_name
        self.modeldir = model_params["dir"]
        self.x_ids = model_params["x_layers"]
        self.y_ids = model_params["y_layers"]

        self.training_loss = model_params["training_loss"]
        self.monitor_loss = model_params["monitor_loss"]
        ### compute combinations
        unique_layer_dims = []
        unique_ydims = []
        # layer_cat = []

        ### TODO - make this not dumb
        for i in range(len(self.layer_dims)):
            if i in self.x_ids and self.layer_dims[i] not in unique_layer_dims:
                unique_layer_dims.append(self.layer_dims[i])
            if i in self.y_ids and self.layer_dims[i] not in unique_ydims:
                unique_ydims.append(self.layer_dims[i])
        sort_unique = list(unique_layer_dims)
        sort_unique.sort()
        unique_freq = [0 for ii in range(len(sort_unique))]
        unique_ydims.sort()
        unique_yfreq = [0 for ii in range(len(unique_ydims))]
        for i in range(len(self.layer_dims)):
            if i in self.x_ids:
                for j in range(len(sort_unique)):
                    if self.layer_dims[i] == sort_unique[j]:
                        unique_freq[j] += 1
            else:
                for j in range(len(unique_ydims)):
                    if self.layer_dims[i] == unique_ydims[j]:
                        unique_yfreq[j] += 1

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], unique_layer_dims,
                                               unique_freq, unique_ydims, unique_yfreq)
        self.base_model.compile(loss=self.training_loss)
        if self.v > 0:
            print(self.base_model.summary())
        callback1 = lossCallback()
        callback2 = ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
                                    monitor = "val_mean_squared_error", verbose=2, mode="min",
                                    save_best_only=True, save_freq="epoch", save_weights_only=True)
        self.callbacks = [callback1]#, callback2]
        print("done setting up")

    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

    def fit(self, train_data, val_data, n_epochs):
        print("fitting...")
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=2)

    def predict(self, val_data):
        ### iterate over y layers
        yhats = [[] for iii in range(len(self.y_ids))]
        ys = [[] for iii in range(len(self.y_ids))]
        for i in range(len(val_data)):
            ### val y at batch i
            ys_i = val_data[i][1]
            y_hats_i = self.base_model(val_data[i][0])
            for j in range(len(self.y_ids)):
                for elty in ys_i[j]:
                    ys[j].append(elty)
                for elth in y_hats_i[j]:
                    yhats[j].append(elth)

        for i in range(len(self.y_ids)):
            yhats[i] = np.array(yhats[i])[:,:,:,0]
            ys[i] = np.array(ys[i])
            print("shape sanity check", i, "-", yhats[i].shape, ys[i].shape)
        self.recent_ys = ys
        self.recent_yhats = yhats
        return ys, yhats

    def save(self):
        self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "wb") as cblog:
            pickle.dump(self.callbacks[0].logs, cblog)

### patch work for transformer
class ArbitraryPatchExtracter(Layer):
    def __init__ (self, patch_size, patch_ul, n_patches, layer_limits):
        self.patch_size = patch_size
        self.patch_ul = patch_ul
        self.n_patches = n_patches

        n_patches_total = n_patches[0] * n_patches[1]
        self.patch_extractor_list = []
        for i in range(n_patches[0]):
            self.patch_extractor_list.append([])
            for j in range(n_patches[1]):
                self.patch_extractor_list[i].append([])
                for k in range(patch_size[0]):
                    for l in range(patch_size[1]):
                        if patch_ul[i]+k < layer_limits[0] and patch_ul[j]+j < layer_limits[1]:
                            self.patch_extractor_list[i][j].append([patch_ul[i]+k, patch_ul[j]+j, 0])
    def call(self, layer):
        ### build list in init
        return tfgather_nd(layer, indices=self.patch_extractor_list * tfshape(layer)[0], batch_dims=1)
        ### something to handle uneven patch sizes due to oob
        ### ...?

class PyramidPatchExtracter(Layer):
    def __init__ (self, layer_dim, n_patches, n_channels):
        super().__init__()
        ### assumption here is that we are already grouping layers by dimension
        ### so this is only working on the n * n * m group of layers of dim n?
        self.layer_dim = layer_dim
        self.n_patches = n_patches
        self.patch_size = math.ceil(self.layer_dim / self.n_patches)
        self.patch_extractor_locs = []
        self.n_channels = n_channels

        ### compute offset ranges
        self.patch_offsets = []
        tm = (self.layer_dim / self.n_patches)
        for i in range(self.n_patches):
            self.patch_offsets.append(math.floor(i * tm))

        ### compute patch offsets
        for i in range(self.n_patches):
            self.patch_extractor_locs.append([])
            for j in range(self.n_patches):
                self.patch_extractor_locs[i].append([])
                for k in range(self.patch_size):
                    for l in range(self.patch_size):
                        self.patch_extractor_locs[i][j].append([self.patch_offsets[i] + k, self.patch_offsets[j] + l])

    def call(self, layer):
        ### build list in init with gather_nd(layer, patch_extractor_list * batches)
        ### reshape to batches, n_patches, n_patches, elts=patch_size^2 * n_channels
        #print("call", tfshape(layer))
        #*int(tfshape(layer)[0])
        return tfmap_fn(fn=lambda intensor: tfreshape(tfgather_nd(intensor, self.patch_extractor_locs),
                                                      (self.n_patches, self.n_patches,
                                                       self.patch_size*self.patch_size*self.n_channels)), elems=layer)
        #return tfreshape(tfgather_nd(layer, indices=[self.patch_extractor_locs], batch_dims=1),
        #                 (-1, self.n_patches, self.n_patches, self.patch_size*self.patch_size*self.n_channels))
        ### something to handle uneven patch sizes due to oob
        ### ... nah?

class Patches(Layer):
    def __init__(self, patch_size, layer_size=None, step_size=None):
        super().__init__()
        self.patch_size = patch_size
        self.layer_size = layer_size
        self.step_size = step_size

    def call(self, images):
        batch_size = tfshape(images)[0]
        patches = tfimage.extract_patches(images=images, sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID",)
        patch_dims = patches.shape[-1]
        patches = tfreshape(patches, [batch_size, -1, patch_dims])
        return patches

### in -- 4d tensor -- [batch, rows, cols, depth]
### out -- 4d tensor -- [batch, patch_i, patch_j, patch_len]
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tfrange(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

### MODEL Transformer 1
class multi_vit:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, n_patches, projection_dim, n_heads, t_layers, mlp_units,
                        output_mode, single_task=False):
        ### split y layers and x layers
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]

        t_units = [projection_dim * 2, projection_dim]

        ### separate input for each layer
        input_layers = [Input(shape=(xdims[ii], xdims[ii], 1)) for ii in range(len(xdims))]

        ### concatenate layers by dim
        unique_xdims = np.unique(xdims)
        xdim_groups = []
        for i in range(len(unique_xdims)):
            xdim_groups.append([])
            for j in range(len(xdims)):
                if xdims[j] == unique_xdims[i]:
                    xdim_groups[i].append(input_layers[j])

        concat_groups = []
        for i in range(len(xdim_groups)):
            concat_groups.append(Concatenate(axis=3)(xdim_groups[i]))

        blocks_out = []

        ### new approach to patch extraction...
        ### step one -- extract patch from each layer
        ### idea here --- do several different patch sizes for variety
        ### reality -- patch sizes
        for j in range(len(n_patches)):
            ### append result of transformer_block to blocks_out
            patches = []
            ### compute patches for each dim group separately, then combine...
            #patchdims = []
            if n_patches[j] > 1:
                for i in range(len(xdim_groups)):
                    patches.append(PyramidPatchExtracter(layer_dim=unique_xdims[i], n_patches=n_patches[j],
                                                n_channels=len(xdim_groups[i]))(concat_groups[i]))
                    #patchdims.append(tfshape(patches[i][3]))
                #print("patch sizes:", patchdims)
            else:
                for i in range(len(xdim_groups)):
                    patches.append(Reshape((-1, 1, 1, unique_xdims[i] * unique_xdims[i] * len*xdim_groups[i]))(xdim_groups[i]))
            ### combine for each dim group...
            patches = Concatenate(axis=3)(patches)

            ### encode patches
            encoded_patches = PatchEncoder(n_patches[j], projection_dim)(patches)

            ### iterate over transformer block layers...
            for _ in range(t_layers):
                ### layer normalization
                x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)

                ### multi-head attn
                attention_output = MultiHeadAttention(num_heads=n_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)

                ### skip connection (1)
                x2 = Add()([attention_output, encoded_patches])
                ### layer normalization (2)
                x3 = LayerNormalization(epsilon=1e-6)(x2)
                ### MLP layer
                for units in t_units:
                    x3 = Dense(units, activation=gelu)(x3)
                    x3 = Dropout(0.1)(x3)
                ### skip connection (3)
                encoded_patches = Add()([x3, x2])
            representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
            representation = Flatten()(representation)
            representation = Dropout(0.5)(representation)

            blocks_out.append(representation)

        ### combine outputs of each separate encoding size...
        working_mdl = tfconcat(blocks_out, 1)

        ### now append the mlp
        # mlp = mlp(rep1, hidden_units, dropout_rate)
        for units in mlp_units:
            working_mdl = Dense(units, activation=gelu)(working_mdl)
            working_mdl = Dropout(0.5)(working_mdl)

        ### it goes... 0: agb 1: wue 2 esi

        ### wue esi agb
        output_layers = []
        if single_task is None:
            if output_mode == "combine":
                output_branch_0 = Dense((ydims[0] ** 2) * 2, activation="relu")(working_mdl)
                output_branch_1 = Dense((ydims[2] ** 2), activation="relu")(working_mdl)

                output_branch_1 = Dense((ydims[2] ** 2), activation="relu")(output_branch_1)
                output_1 = Reshape((ydims[2], ydims[2]), name="out_"+str(ydims[2]))(output_branch_1)

                output_branch_0 = Dense((ydims[0] ** 2) * 2, activation="relu")(output_branch_0)
                output_branch_0 = Concatenate([output_branch_0, output_branch_1])
                output_branch_0 = Dense((ydims[0] ** 2) * 2 + (ydims[2] ** 2), activation="relu")(output_branch_0)

                output_0a = Dense(ydims[0] ** 2, activation="relu")(output_branch_0)
                output_0b = Dense(ydims[1] ** 2, activation="relu")(output_branch_1)

                output_0a = Reshape((ydims[0], ydims[0]), name="out_"+str(ydims[0]) + "_0")(output_0a)
                output_0b = Reshape((ydims[1], ydims[1]), name="out_" + str(ydims[1]) + "_1")(output_0b)

                output_layers = [output_0a, output_0b, output_1]
            elif output_mode == "inverse":
                output_branch_0 = Dense((ydims[0] ** 2) * 2, activation="relu")(working_mdl)
                output_branch_1 = Dense((ydims[2] ** 2), activation="relu")(working_mdl)

                output_branch_1 = Dense((ydims[2] ** 2), activation="relu")(output_branch_1)
                output_1 = Reshape((ydims[2], ydims[2]), name="out_" + str(ydims[2]))(output_branch_1)

                output_branch_0 = Dense((ydims[0] ** 2) * 2, activation="relu")(output_branch_0)

                output_0a = Dense(ydims[0] ** 2, activation="relu")(output_branch_0)
                output_0b = Dense(ydims[1] ** 2, activation="relu")(output_branch_1)

                output_0a = Reshape((ydims[0], ydims[0]), name="out_" + str(ydims[0]) + "_0")(output_0a)
                output_0b = Reshape((ydims[1], ydims[1]), name="out_" + str(ydims[1]) + "_1")(output_0b)

                output_layers = [output_0a, output_0b, output_1]
            elif output_mode == "separate":
                output_branch_0 = Dense((ydims[0] ** 2), activation="relu")(working_mdl)
                output_branch_1 = Dense((ydims[1] ** 2), activation="relu")(working_mdl)
                output_branch_2 = Dense((ydims[2] ** 2), activation="relu")(working_mdl)

                output_branch_0 = Dense((ydims[0] ** 2), activation="relu")(output_branch_1)
                output_branch_1 = Dense((ydims[1] ** 2), activation="relu")(output_branch_1)
                output_branch_2 = Dense((ydims[2] ** 2), activation="relu")(output_branch_1)

                output_0 = Reshape((ydims[0], ydims[0]), name="out_" + str(ydims[0]) + "_0")(output_branch_0)
                output_1 = Reshape((ydims[1], ydims[1]), name="out_" + str(ydims[1]) + "_1")(output_branch_1)
                output_2 = Reshape((ydims[2], ydims[2]), name="out_" + str(ydims[2]) + "_2")(output_branch_2)

                output_layers = [output_0, output_1, output_2]
        else:
            output_single = Dense((ydims[single_task] ** 2), activation="relu")(working_mdl)
            output_single = Dense((ydims[single_task] ** 2), activation="relu")(output_single)
            output_single = Reshape((ydims[single_task], ydims[single_task]), name="out_" + str(ydims[single_task]))(output_single)

            output_layers = output_single

        print("finished setting up")
        return Model(inputs=input_layers, outputs=output_layers)

    def setup(self, model_params, model_name):
        self.name = model_name
        self.x_ids = model_params["x_layers"]
        self.y_ids = model_params["y_layers"]
        self.layer_dims = model_params["layerdims"]
        self.modeldir = model_params["dir"]
        self.v = model_params["verbosity"]

        self.training_loss = model_params["training_loss"]
        self.monitor_loss = model_params["monitor_loss"]

        ### specific params
        self.n_patches = model_params["n_patches"]
        self.projection_dim = model_params["proj_dim"]
        self.n_heads = model_params["n_heads"]
        self.t_layers = model_params["n_heads"]
        self.mlp_units = model_params["mlp_units"]
        self.output_mode = model_params["output_mode"]
        self.singletask=model_params["singletask"]

        print(self.x_ids, self.y_ids)
        #n_patches, projection_dim, n_heads, t_layers, mlp_units, output_mode, single_task=False
        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], n_patches=self.n_patches,
                                               projection_dim=self.projection_dim, n_heads=self.n_heads,
                                               t_layers=self.t_layers, mlp_units=self.mlp_units,
                                               output_mode=self.output_mode, single_task=self.singletask)
        print("compiling...")
        self.base_model.compile(loss=self.training_loss)
        if self.v > 0:
            print(self.base_model.summary())
        callback1 = lossCallback()
        callback2 = ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
                                    monitor = "val_mean_squared_error", verbose=2, mode="min",
                                    save_best_only=True, save_freq="epoch", save_weights_only=True)
        self.callbacks = [callback1, callback2]
        print("done setting up transformer")

    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

    def fit(self, train_data, val_data, n_epochs):
        if self.singletask is not None:
            train_data.set_single_y(self.singletask)
            val_data.set_single_y(self.singletask)
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=self.v)
        if self.singletask is not None:
            train_data.set_multi_y()
            val_data.set_multi_y()

    def predict(self, val_data):
        if self.singletask is not None:
            val_data.set_single_y(self.singletask)
        yhats = [[] for iii in range(len(val_data.use_y_ids))]
        ys = [[] for iii in range(len(val_data.use_y_ids))]
        ### iterate over batches?
        for i in range(len(val_data)):
            ### val y at batch i
            ys_i = val_data[i][1]
            y_hats_i = self.base_model(val_data[i][0])
            if self.singletask is not None:
                y_hats_i = [y_hats_i]
                #print(y_hats_i.shape)
            ### iterate over y layers
            for j in range(len(val_data.use_y_ids)):
                for elty in ys_i[j]:
                    ys[j].append(elty)
                for elth in y_hats_i[j]:
                    yhats[j].append(elth)

        print(yhats[0][0].shape)
        print(len(yhats))
        print(len(yhats[0]))

        for i in range(len(val_data.use_y_ids)):
            yhats[i] = np.array(yhats[i])[:, :, :]#yhats[i] = np.array(yhats[i])[:, :, :, 0]
            ys[i] = np.array(ys[i])
            print("shape sanity check", i, "-", yhats[i].shape, ys[i].shape)

        if self.singletask is not None:
            val_data.set_multi_y()

        self.recent_ys = ys
        self.recent_yhats = yhats
        return ys, yhats

    def save(self):
        self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "wb") as cblog:
            pickle.dump(self.callbacks[0].logs, cblog)