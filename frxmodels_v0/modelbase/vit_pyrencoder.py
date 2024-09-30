import numpy as np

from tensorflow import keras
from tensorflow import shape as tfshape
from tensorflow import image as tfimage
from tensorflow import reshape as tfreshape
from tensorflow import range as tfrange
from tensorflow import concat as tfconcat
from tensorflow import constant as tfconstant
from tensorflow import map_fn as tfmap_fn
from tensorflow import gather_nd as tfgather_nd
from tensorflow.nn import gelu

from modelbase import mltools as mlt

import math
import pickle
import time

### pyramid patch extractor
### patch work for transformer
class ArbitraryPatchExtracter(keras.layers.Layer):
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

class PyramidPatchExtracter(keras.layers.Layer):
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

class Patches(keras.layers.Layer):
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
class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tfrange(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

### MODEL Transformer 1
class multi_vit:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, n_patches, projection_dim, n_heads, t_layers, mlp_units,
                        output_mode, single_task=False, pdims_out=None, y_layer_names=None):
        ### split y layers and x layers
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]

        t_units = [projection_dim * 2, projection_dim]

        ### separate input for each layer
        input_layers = [keras.layers.Input(shape=(xdims[ii], xdims[ii], 1)) for ii in range(len(xdims))]

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
            concat_groups.append(keras.layers.Concatenate(axis=3)(xdim_groups[i]))

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
                    patches.append(keras.layers.Reshape((-1, 1, 1, unique_xdims[i] * unique_xdims[i] * len*xdim_groups[i]))(xdim_groups[i]))
            ### combine for each dim group...
            patches = keras.layers.Concatenate(axis=3)(patches)

            ### encode patches
            encoded_patches = PatchEncoder(n_patches[j], projection_dim)(patches)

            ### iterate over transformer block layers...
            for _ in range(t_layers):
                ### layer normalization
                x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

                ### multi-head attn
                attention_output = keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)

                ### skip connection (1)
                x2 = keras.layers.Add()([attention_output, encoded_patches])
                ### layer normalization (2)
                x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
                ### MLP layer
                for units in t_units:
                    x3 = keras.layers.Dense(units, activation=gelu)(x3)
                    x3 = keras.layers.Dropout(0.1)(x3)
                ### skip connection (3)
                encoded_patches = keras.layers.Add()([x3, x2])
            representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            representation = keras.layers.Flatten()(representation)
            representation = keras.layers.Dropout(0.5)(representation)

            blocks_out.append(representation)

        ### combine outputs of each separate encoding size...
        working_mdl = keras.layers.Concatenate(axis=1)(blocks_out)

        ### now append the mlp
        # mlp = mlp(rep1, hidden_units, dropout_rate)
        fc = mlt.dense_block(working_mdl, mlp_units)

        unique_layer_dims, y_unique, unique_freq, y_freq = pdims_out

        y_split_out = mlt.y_block(fc, ydims, y_unique, y_freq, y_layer_names,
                                  output_mode, single_task)
        ### it goes... 0: agb 1: wue 2 esi

        ### wue esi agb


        print("finished setting up")
        return keras.models.Model(inputs=input_layers, outputs=y_split_out)

    def setup(self, model_params, model_dir, model_name, verbosity, cb_params):
        self.name = model_name
        self.x_ids = model_params["x_layers"]
        self.y_ids = model_params["y_layers"]
        self.layer_dims = model_params["layer_dims"]
        self.modeldir = model_dir
        self.v = verbosity

        self.training_loss = model_params["training_loss"]
        self.monitor_loss = model_params["monitor_loss"]

        ### specific params
        self.n_patches = model_params["n_patches"]
        self.projection_dim = model_params["proj_dim"]
        self.n_heads = model_params["n_heads"]
        self.t_layers = model_params["n_heads"]
        self.mlp_units = model_params["dense_layers"]
        self.output_mode = model_params["output_block"]
        self.singletask = model_params["singletask"]

        y_layer_names = model_params["layer_names"][len(self.x_ids):]

        pdims_out = mlt.process_dims(self.layer_dims, self.x_ids, self.y_ids)

        print(self.x_ids, self.y_ids)
        #n_patches, projection_dim, n_heads, t_layers, mlp_units, output_mode, single_task=False
        self.base_model = self.make_base_model(self.layer_dims, len(self.x_ids), n_patches=self.n_patches,
                                               projection_dim=self.projection_dim, n_heads=self.n_heads,
                                               t_layers=self.t_layers, mlp_units=self.mlp_units,
                                               output_mode=self.output_mode, single_task=self.singletask,
                                               pdims_out=pdims_out, y_layer_names=y_layer_names)
        print("compiling...")
        opt = keras.optimizers.Adam(learning_rate=model_params["learning_rate"])
        self.base_model.compile(loss={'ECOSTRESSWUE':  keras.losses.MeanSquaredError(), 
                                      'ECOSTRESSESI':  keras.losses.MeanSquaredError(),
                                      'GEDIAGB': keras.losses.MeanSquaredError()},
                                loss_weights={'ECOSTRESSWUE': 1.0, 
                                              'ECOSTRESSESI': 1.0,
                                              'GEDIAGB': 1.0},
                                optimizer=opt,
                                metrics=[keras.metrics.MeanSquaredError(), #tf.keras.metrics.MeanAbsoluteError()])
                                         keras.metrics.MeanSquaredError(),
                                         keras.metrics.MeanSquaredError()])
        if self.v > 0:
            print(self.base_model.summary())
        self.callbacks = []
        for cb_name in cb_params:
            if cb_name == "loss":
                self.callbacks.append(mlt.lossCallback())
            elif cb_name == "checkpoint":
                self.callbacks.append(
                    keras.callbacks.ModelCheckpoint(self.modeldir + "/checkpoint_" + self.name + ".weights.h5",
                                                       monitor="val_loss",
                                                       verbose=self.v, mode="min",
                                                       save_best_only=True, save_freq="epoch",
                                                       save_weights_only=True))
        print("done setting up transformer")


    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".weights.h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

    #train_wrangler, val_wrangler, train_params["n_epochs_default"], train_params["workers"], train_params["multip"]
    def fit(self, train_data, val_data, n_epochs, n_workers, multip):
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
                # print(y_hats_i.shape)
            ### iterate over y layers
            for j in range(len(val_data.use_y_ids)):
                for elty in ys_i[j]:
                    ys[j].append(elty)
                for elth in y_hats_i[j]:
                    yhats[j].append(elth)

        for i in range(len(val_data.use_y_ids)):
            yhats[i] = np.array(yhats[i])
            ys[i] = np.array(ys[i])
            print("shape sanity check", i, "-", yhats[i].shape, ys[i].shape)

        if self.singletask is not None:
            val_data.set_multi_y()

        self.recent_ys = ys
        self.recent_yhats = yhats
        return ys, yhats

    def save(self):
        self.base_model.save_weights(self.modeldir + "/model_" + self.name + ".weights.h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "wb") as cblog:
            pickle.dump(self.callbacks[0].logs, cblog)