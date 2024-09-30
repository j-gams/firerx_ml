### cascade mid
import tensorflow as tf
import numpy as np
import pickle
import math
from modelbase import mltools as mlt

class model_cascade2_mid:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, layer_dims, y_break, unique_dims_sorted, frequency, y_unique, y_frequency, model_params):
        xdims = layer_dims[:y_break]
        ydims = layer_dims[y_break:]
        inputs = []
        convs1 = [[] for ii in range(len(unique_dims_sorted))]
        convs2 = []

        if self.v != 0:
            print("building cascade2_mid model")
            print(unique_dims_sorted)
            print(frequency)

        conv_dims = list(unique_dims_sorted)
        sfreq = list(frequency)

        ### step 1 - gather layers into groups
        for i in range(len(xdims)):
            inputs.append(tf.keras.layers.Input(shape=(xdims[i], xdims[i], 1)))
            for j in range(len(unique_dims_sorted)):
                if xdims[i] == unique_dims_sorted[j]:
                    convs1[j].append(inputs[i])
        ### step 2 - concatenate inputs in same groups
        for j in range(len(unique_dims_sorted)):
            convs2.append(tf.keras.layers.Concatenate(axis=3)(convs1[j]))

        ### step 3 - action
        for j in range(len(conv_dims)):
            if unique_dims_sorted[j] == 1:
                ### 1 -> 2
                convs2[j] = tf.keras.layers.Conv2DTranspose(filters=8 * frequency[j], kernel_size=(2, 2),
                                            strides=(1, 1))(convs2[j])
                conv_dims[j] = 2
            elif unique_dims_sorted[j] == 2:
                ### 2-> 2
                convs2[j] = tf.keras.layers.Conv2D(filters=8 * frequency[j], kernel_size=(1, 1), strides=(1, 1),
                                   padding="same")(convs2[j])
            else:
                ### 1st conv
                ### 34 -> 32
                ### or 27 -> 25
                convs2[j] = tf.keras.layers.Conv2D(filters=8 * frequency[j], kernel_size=(3, 3), strides=(1, 1),
                                   padding="valid")(convs2[j])
                conv_dims[j] -= 2

        ### step 4
        for j in range(len(conv_dims)):
            if conv_dims[j] <= 2:
                # convs2[j] = Concatenate(axis=3)(convs2[j])
                pass
            else:
                ### 1st maxpool
                ### 32 -> 16
                ### or 25 -> 13
                convs2[j] = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                conv_dims[j] = math.ceil(conv_dims[j] / 2)

        ### step 5
        for j in range(len(conv_dims)):
            if conv_dims[j] > 2:
                ### 2nd conv
                ### 16 -> 8 and 13 -> 7
                convs2[j] = tf.keras.layers.Conv2D(filters=16 * frequency[j], kernel_size=(3, 3), strides=(2, 2),
                                   padding="same")(convs2[j])
                conv_dims[j] = math.ceil(conv_dims[j] / 2)

            ### step 6
            if conv_dims[j] > 2:
                ### 2nd maxpool
                ### 8 -> 4 and 7 -> 4
                convs2[j] = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(convs2[j])
                conv_dims[j] = math.ceil(conv_dims[j] / 2)

        ### step 7
        for j in range(len(conv_dims)):
            if conv_dims[j] > 2:
                ### 3rd conv
                ### 4 -> 2 and 4 -> 2
                convs2[j] = tf.keras.layers.Conv2D(filters=16 * frequency[j], kernel_size=(3, 3), strides=(2, 2),
                                   padding="same")(convs2[j])
                conv_dims[j] = math.ceil(conv_dims[j] / 2)

        ### step 9
        """if len(conv_dims) > 1:
            convdims = [2, 2, 2]
            convs2[0] = [convs2[0], convs2[1], convs2[2]]
            sfreq[0] = frequency[0] + frequency[1] + frequency[2]
            convs2[0] = tf.keras.layers.Concatenate(axis=3)(convs2[0])"""

        t_conv_dims = []
        t_convs2 = []
        t_freq = []
        for i in range(len(conv_dims)):
            if conv_dims[i] not in t_conv_dims:
                t_conv_dims.append(conv_dims[i])
                t_convs2.append([convs2[i]])
                t_freq.append(frequency[i])
            else:
                t_convs2[-1].append(convs2[i])
                t_freq[-1] += frequency[i]
        conv_dims = t_conv_dims
        convs2 = t_convs2
        sfreq = t_freq

        ### step 8 -- concat
        for j in range(len(conv_dims)):
            convs2[j] = tf.keras.layers.Concatenate(axis=3)(convs2[j])

        ### step 11 conv and flatten
        convs2[0] = tf.keras.layers.Conv2D(filters=16 * len(xdims), kernel_size=(2, 2), strides=(2, 2),
                           padding="same")(convs2[0])

        fc = tf.keras.layers.Flatten()(convs2[0])

        ### step 12 -- dense layers
        fc = mlt.dense_block(fc, model_params["dense_layers"])

        ### step 13 -- output block
        ### input_layer, y_dims, y_unique, y_frequency, y_names, block_version, single_task
        y_split_out = mlt.y_block(fc, ydims, y_unique, y_frequency, model_params["layer_names"][y_break:],
                                  model_params["output_block"], model_params["single_task"])

        if self.v != 0:
            print("finished setting up")
        return tf.keras.models.Model(inputs=inputs, outputs=y_split_out)


    def setup(self, model_params, model_dir, model_name, verbosity, cb_params):
        self.v = verbosity
        self.layer_dims = model_params["layer_dims"]
        self.name = model_name
        self.modeldir = model_dir
        self.x_ids = model_params["x_layers"]
        self.y_ids = model_params["y_layers"]
        self.training_loss = model_params["training_loss"]
        self.monitor_loss = model_params["monitor_loss"]
        self.singletask = model_params["single_task"]


        unique_layer_dims, y_unique, unique_freq, y_freq = mlt.process_dims(self.layer_dims, self.x_ids, self.y_ids)

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], unique_layer_dims,
                                               unique_freq, y_unique, y_freq, model_params)
        opt = tf.keras.optimizers.Adam(learning_rate=model_params["learning_rate"])
        #self.base_model.compile(loss=self.training_loss, optimizer=opt)
        self.base_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                                metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
        if self.v > 0:
            print(self.base_model.summary())

        if self.v > 0:
            print("done setting up")

        ### TODO -- get callbacks...
        self.callbacks = []
        for cb_name in cb_params:
            if cb_name == "loss":
                self.callbacks.append(mlt.lossCallback())
            elif cb_name == "checkpoint":
                self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(self.modeldir + "/checkpoint_" + self.name + ".h5",
                                                                 monitor="val_loss",
                                                                 verbose=self.v, mode="min",
                                                                 save_best_only=True, save_freq="epoch",
                                                                 save_weights_only=True))

    def load(self):
            self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
            with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
                picklelog = pickle.load(cblog)
            self.callbacks[0].resume_from_load(picklelog)

    def fit(self, train_data, val_data, n_epochs, n_workers, multip):
        if self.singletask is not None:
            train_data.set_single_y(self.singletask)
            val_data.set_single_y(self.singletask)
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=self.v, workers=n_workers, use_multiprocessing=multip)
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