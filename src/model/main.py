from model import Model
import numpy as np
import os

ASSETS_DIR = "../../assets"
DATA_DIR = os.path.join(ASSETS_DIR, "data")
MODEL_DIR = os.path.join(ASSETS_DIR, "model")

def main():
    x_train = np.load(os.path.join(DATA_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(DATA_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    hyperparams = {
        'learning_rate': 0.0001,
        'batch_size': 10,
        'epochs': 10,
        'validation_data': (x_test, y_test)
    }

    model = Model(x_train[0].shape, y_train[0].shape, hyperparams)

    model.fit(x_train, y_train, hyperparams)
    model.save(os.path.join(MODEL_DIR, "model.h5"))

if __name__ == '__main__':
    main()