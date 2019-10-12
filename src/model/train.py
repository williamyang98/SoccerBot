from model import Model
import numpy as np
import os
from paths import DATA_DIR, MODEL_DIR
from evaluation import calculate_IOU, calculate_loss

def main():
    x_train = np.load(os.path.join(DATA_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(DATA_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    hyperparams = {
        'learning_rate': 0.0001,
        'batch_size': 100,
        'epochs': 10,
        'loss': calculate_loss,
        'metrics': [calculate_IOU, "accuracy"],
        'validation_data': (x_test, y_test)
    }

    model = Model((256,256,3), (4,), hyperparams)
    model.load(os.path.join(MODEL_DIR, "model.h5"))
    model.summary()

    model.fit(x_train, y_train, hyperparams)
    model.save(os.path.join(MODEL_DIR, "model.h5"))

if __name__ == '__main__':
    main()
