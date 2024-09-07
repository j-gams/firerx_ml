import tensorflow as tf
import numpy as np
import pickle
from modelbase import mltools as mlt

### MODEL: Flat 2
class model_flat2:
    def __init__(self):
        self.base_model = None

    def make_base_model(self, ldims, yoff, uniquedimssorted, freq, ydims_unique, ydims_freq, model_params):
        xdims = ldims[:yoff]
        ydims = ldims[yoff:]
        inputs = []
        convs = []
        for i in range(len(xdims)):
            inputs.append(tf.keras.layers.Input(shape=(xdims[i], xdims[i], 1)))

        c1 = tf.keras.layers.Concatenate(axis=3)(inputs)
        ### 34 -> 32 or 27 -> 25
        c1 = tf.keras.layers.Conv2D(filters=30, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 32 -> 30 or 25 -> 23
        c1 = tf.keras.layers.Conv2D(filters=60, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 30 -> 28 or 23 -> 21
        c1 = tf.keras.layers.Conv2D(filters=157, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 30 -> 15 or 23 -> 12
        c1 = tf.keras.layers.Conv2D(filters=304, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        ### 15 -> 8 or 12 -> 6
        c1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)

        ### 8 -> 4 or 6 -> 3
        c1 = tf.keras.layers.Conv2D(filters=304, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        ### 4 -> 2 or 3 -> 2
        c1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)
        merge = tf.keras.layers.Flatten()(c1)
        ### step 12 -- dense layers
        merge = mlt.dense_block(merge, model_params["dense_layers"])
        print(ydims, ydims_unique, ydims_freq)
        ### step 13 -- output block
        ### input_layer, y_dims, y_unique, y_frequency, y_names, block_version, single_task
        y_split_out = mlt.y_block(merge, ydims, ydims_unique, ydims_freq, model_params["layer_names"][yoff:],
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
        ### compute combinations
        unique_layer_dims = []
        unique_ydims = []
        # layer_cat = []

        unique_layer_dims, y_unique, unique_freq, y_freq = mlt.process_dims(self.layer_dims, self.x_ids, self.y_ids)

        self.base_model = self.make_base_model(self.layer_dims, self.y_ids[0], unique_layer_dims,
                                               unique_freq, y_unique, y_freq, model_params)
        opt = tf.keras.optimizers.Adam(learning_rate=model_params["learning_rate"])
        self.base_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,
                                metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])

        if self.v > 0:
            print(self.base_model.summary())
        self.callbacks = []
        for cb_name in cb_params:
            if cb_name == "loss":
                self.callbacks.append(mlt.lossCallback())
            elif cb_name == "checkpoint":
                self.callbacks.append(
                    tf.keras.callbacks.ModelCheckpoint(self.modeldir + "/checkpoint_" + self.name + ".h5",
                                                       monitor="val_loss",
                                                       verbose=self.v, mode="min",
                                                       save_best_only=True, save_freq="epoch",
                                                       save_weights_only=True))
        if self.v > 0:
            print("done setting up")

    def load(self):
        self.base_model.load_weights(self.modeldir + "/model_" + self.name + ".h5")
        with open(self.modeldir + "/cblog_" + self.name + ".txt", "rb") as cblog:
            picklelog = pickle.load(cblog)
        self.callbacks[0].resume_from_load(picklelog)

    def fit(self, train_data, val_data, n_epochs, n_workers, multip):
        print("fitting...")
        self.base_model.fit(train_data, callbacks=self.callbacks, epochs=n_epochs, validation_data=val_data,
                            verbose=self.v, workers=n_workers, use_multiprocessing=multip)

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
