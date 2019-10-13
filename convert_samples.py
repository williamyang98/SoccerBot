import glob
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import random

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
    x, y = get_training_data(image_filepaths, args.labels_dir)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.ratio)

    # save to file
    np.save(os.path.join(args.output_dir, "x_train.npy"), x_train)
    np.save(os.path.join(args.output_dir, "x_test.npy"), x_test)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)

def get_training_data(image_filepaths, labels_dir):
    x_train = []
    y_train = []
    total = len(image_filepaths)
    for count, image_filepath in enumerate(image_filepaths):
        if (count+1) % 10 == 0:
            print("\r{0}/{1}".format(count+1, total), end='') 
        try:
            image, bounding_box = get_converted_sample(image_filepath, labels_dir)
            x_train.append(image)
            y_train.append(bounding_box)
        except IOError as ex:
            print("Couldn't open file: {0}".format(ex.filename))
            continue

    return (np.array(x_train), np.array(y_train))

# returns image as numpy array, and labels as bounding box
def get_converted_sample(image_filepath, labels_dir):
    image = get_image(image_filepath)
    filename = os.path.basename(image_filepath).split('.')[0]
    label_filepath = os.path.join(labels_dir, "{0}.txt".format(filename))
    bounding_box = get_label(label_filepath)
    return (image, bounding_box)

def get_label(filepath):
    with open(filepath, "r") as file:
        for line in file.readlines():
            id, x_centre, y_centre, width, height = map(float, line.strip().split(' '))
            label = (x_centre, y_centre, width, height)
            return label

def get_image(filepath):
    with Image.open(filepath, mode='r') as img:
        pixel_data = np.asarray(img) / 255 # normalise
    return pixel_data

if __name__ == '__main__':
    main()