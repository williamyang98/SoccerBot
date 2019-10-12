import glob
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import random

ASSETS_DIR = "../../assets/"
DEFAULT_IMAGE_DIR = os.path.join(ASSETS_DIR, "samples", "*JPEG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--save-dir", type=str, default=os.path.join(ASSETS_DIR, "data"))
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dir", type=str, default=DEFAULT_IMAGE_DIR)
    args = parser.parse_args()
    # load files
    files = glob.glob(args.dir)
    if args.shuffle:
        random.shuffle(files)
    files = files[:args.max_samples]
    x, y = get_training_data(files)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.ratio)
    # save to file
    np.save(os.path.join(args.save_dir, "x_train.npy"), x_train)
    np.save(os.path.join(args.save_dir, "x_test.npy"), x_test)
    np.save(os.path.join(args.save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.save_dir, "y_test.npy"), y_test)

def get_training_data(image_files):
    x_train = []
    y_train = []
    total = len(image_files)
    for count, image_file in enumerate(image_files):
        if (count+1) % 10 == 0:
            print("\r{0}/{1}".format(count+1, total), end='') 

        filename = os.path.basename(image_file).split('.')[0]
        label_file = os.path.join(ASSETS_DIR, "labels", "{0}.txt".format(filename))
        # process
        try:
            pixel_data = get_pixel_data(image_file)
            labels = get_labels(label_file)
            x_train.append(pixel_data)
            y_train.append(labels)
        except IOError as ex:
            print("Couldn't open file: {0}".format(ex.filename))
            continue
    
    return (np.array(x_train), np.array(y_train))

def get_labels(filepath):
    with open(filepath, "r") as file:
        for line in file.readlines():
            id, x_centre, y_centre, width, height = map(float, line.strip().split(' '))
            label = (x_centre, y_centre, width, height)
            return label

def get_pixel_data(filepath):
    with Image.open(filepath, mode='r') as img:
        pixel_data = np.asarray(img) / 255 # normalise
    return pixel_data

if __name__ == '__main__':
    main()

    



