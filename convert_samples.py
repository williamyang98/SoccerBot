import glob
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import random

from src.converter import Converter

ASSETS_DIR = "assets/"
IMAGES_DIR = os.path.join(ASSETS_DIR, "data", "samples", "*JPEG")
LABELS_DIR = os.path.join(ASSETS_DIR, "data", "labels")
OUTPUT_DIR = os.path.join(ASSETS_DIR, "data", "compressed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--images-dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--labels-dir", type=str, default=LABELS_DIR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    # load files
    image_filepaths = glob.glob(args.images_dir)
    if not args.no_shuffle:
        random.shuffle(image_filepaths)
    image_filepaths = image_filepaths[:args.max_samples]

    # get data
    converter = Converter()
    x, y = converter.get_training_data(image_filepaths, args.labels_dir)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.ratio)

    # save to file
    np.save(os.path.join(args.output_dir, "x_train.npy"), x_train)
    np.save(os.path.join(args.output_dir, "x_test.npy"), x_test)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)



if __name__ == '__main__':
    main()