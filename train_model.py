import numpy as np
import os
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

from src.model import Model

MODEL_DIR = "assets/model/"
DATA_DIR = "assets/data/"
LABELS_FILE = os.path.join(DATA_DIR, "labels.txt")
IMAGES_DIR = os.path.join(DATA_DIR, "samples")

TARGET_SIZE = (160, 227)
OUTPUT_SIZE = 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", default=os.path.join(MODEL_DIR, "model.h5"))
    parser.add_argument("--model-out", default=os.path.join(MODEL_DIR, "model.h5"))
    parser.add_argument("--labels", default=LABELS_FILE)
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    # hypeparams
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.15)

    args = parser.parse_args()

    image_gen = ImageDataGenerator(validation_split=args.ratio, rescale=1.0/255.0)
    dataframe = pd.read_csv(args.labels, delim_whitespace=True)
    label_names = ["x_centre", "y_centre", "width", "height", "confidence"]


    training_generator = image_gen.flow_from_dataframe(
        dataframe=dataframe,
        directory=args.images_dir,
        x_col="filename",
        y_col=label_names,
        subset="training",
        batch_size=args.batch_size,
        class_mode="raw",
        target_size=TARGET_SIZE)

    validation_generator = image_gen.flow_from_dataframe(
        dataframe=dataframe,
        directory=args.images_dir,
        x_col="filename",
        y_col=label_names,
        subset="validation",
        batch_size=args.batch_size,
        class_mode="raw",
        target_size=TARGET_SIZE)

    hyperparams = {
        'learning_rate': args.learning_rate,
    }

    model = Model(TARGET_SIZE+(3,), (OUTPUT_SIZE,), hyperparams)

    try:
        model.load(args.model_in)
    except IOError:
        print("Unable to load model: {0}".format(args.model_in))

    model.summary()

    steps_per_epoch = training_generator.n//training_generator.batch_size
    validation_steps = validation_generator.n//validation_generator.batch_size

    print("steps per epoch: {0}".format(steps_per_epoch))
    print("validation steps {0}".format(validation_steps))

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=args.epochs,
    )
    model.save(args.model_out)

if __name__ == '__main__':
    main()
