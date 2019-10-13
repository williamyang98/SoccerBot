import numpy as np
import os
import argparse

from src.model import Model

MODEL_DIR = "assets/model/"
DATA_DIR = "assets/data/compressed"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", default=os.path.join(MODEL_DIR, "model.h5"))
    parser.add_argument("--model-out", default=os.path.join(MODEL_DIR, "model.h5"))
    parser.add_argument("--training-data-dir", default=DATA_DIR)
    # hypeparams
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    x_train = np.load(os.path.join(args.training_data_dir, "x_train.npy"))
    y_train = np.load(os.path.join(args.training_data_dir, "y_train.npy"))
    x_test = np.load(os.path.join(args.training_data_dir, "x_test.npy"))
    y_test = np.load(os.path.join(args.training_data_dir, "y_test.npy"))

    hyperparams = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'validation_data': (x_test, y_test)
    }

    model = Model((256,256,3), (4,), hyperparams)
    try:
        model.load(args.model_in)
    except IOError:
        print("Unable to load model: {0}".format(args.model_in))

    model.summary()

    model.fit(x_train, y_train, hyperparams)
    model.evaluate(x_test, y_test)
    model.save(args.model_out)

if __name__ == '__main__':
    main()
