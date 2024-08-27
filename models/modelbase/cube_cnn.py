from tensorflow import keras
import numpy as np
import pickle

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
            inputs.append(keras.models.Input(shape=(xdims[i], xdims[i], 1)))

        c1 = keras.models.Concatenate(axis=3)(inputs)
        ### 34 -> 32 or 27 -> 25
        c1 = keras.models.Conv2D(filters=40, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 32 -> 30 or 25 -> 23
        c1 = keras.models.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 30 -> 28 or 23 -> 21
        c1 = keras.models.Conv2D(filters=157, kernel_size=(3, 3), strides=(1, 1), padding="valid")(c1)
        ### 30 -> 15 or 23 -> 12
        c1 = keras.models.Conv2D(filters=304, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        ### 15 -> 8 or 12 -> 6
        c1 = keras.models.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)

        ### 8 -> 4 or 6 -> 3
        c1 = keras.models.Conv2D(filters=304, kernel_size=(3, 3), strides=(2, 2), padding="same")(c1)
        ### 4 -> 2 or 3 -> 2
        c1 = keras.models.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(c1)
        merge = keras.models.Flatten()(c1)
        c2 = keras.models.Dense(1600, activation="relu")(merge)
        c2 = keras.models.Dense(1800, activation="relu")(c2)
        c2 = keras.models.Dense(1600, activation="relu")(c2)

        y1a = keras.models.Dense((1 ** 2) * 1, activation="relu")(c2)
        y2a = keras.models.Dense((15 ** 2) * 2, activation="relu")(c2)

        y1b = keras.models.Reshape((1, 1, 1), name="agb")(y1a)
        y2b = keras.models.Dense(15 ** 2, activation="relu")(y2a)
        y2c = keras.models.Reshape((15, 15, 1), name="wue")(y2b)
        y2d = keras.models.Dense(15 ** 2, activation="relu")(y2a)
        y2e = keras.models.Reshape((15, 15, 1), name="esi")(y2d)
        yfinal = [y2c, y2e, y1b]

        return keras.models.Model(inputs=inputs, outputs=yfinal)

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
        callback1 = keras.models.lossCallback()
        callback2 = keras.models.ModelCheckpoint(self.modeldir + "/chkpt_" + self.name + ".h5",
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
