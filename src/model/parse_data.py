import glob
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

ASSETS_DIR = "../../assets/"

image_files = glob.glob(os.path.join(ASSETS_DIR, "samples", "*JPEG"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--save-dir", type=str, default=os.path.join(ASSETS_DIR, "data"))
    args = parser.parse_args()

    x, y = get_training_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.ratio)
    # save to file
    np.save(os.path.join(args.save_dir, "x_train.npy"), x_train)
    np.save(os.path.join(args.save_dir, "x_test.npy"), x_test)
    np.save(os.path.join(args.save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.save_dir, "y_test.npy"), y_test)

def get_training_data():
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
    labels = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            id, x_centre, y_centre, width, height = map(float, line.strip().split(' '))
            label = (x_centre, y_centre, width, height)
            labels.append(np.asarray(label))
    return labels

def get_pixel_data(filepath):
    img = Image.open(filepath, mode='r')
    pixel_data = np.asarray(img) / 255 # normalise
    return pixel_data

if __name__ == '__main__':
    main()

    



