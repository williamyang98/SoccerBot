import numpy as np
import os
import argparse

from model import Model
from paths import DATA_DIR, MODEL_DIR
from evaluation import calculate_IOU, calculate_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", default=os.path.join(MODEL_DIR, "model.h5"))
    parser.add_argument("--model-out", default=os.path.join(MODEL_DIR, "model.h5"))

    args = parser.parse_args()

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
    try:
        model.load(
            args.model_in,
            {'calculate_loss': calculate_loss, 'calculate_IOU': calculate_IOU})
        # model.load_weights(os.path.join(MODEL_DIR, "model-weights.h5"))
    except IOError:
        print("Unable to load model: {0}".format())

    model.summary()

    model.fit(x_train, y_train, hyperparams)
    model.evaluate(x_test, y_test)
    model.save(args.model_out)

if __name__ == '__main__':
    main()
