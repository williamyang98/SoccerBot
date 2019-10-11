import glob
import os
from PIL import Image
import numpy as np

ASSETS_DIR = "../../assets/"

image_files = glob.glob(os.path.join(ASSETS_DIR, "samples", "*JPEG"))

def get_training_data():
    x_train = []
    y_train = []
    for image_file in image_files:
        filename = os.path.basename(image_file).split('.')[0]
        # ignore if no labels
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
    
    return (x_train, y_train)

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

    



