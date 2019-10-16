import numpy as np
import cv2
import os
from PIL import Image

class Converter:
    def __init__(self, output_size=(256,256)):
        self.output_size = output_size

    def get_training_data(self, image_filepaths, labels_dir):
        x_train = []
        y_train = []
        total = len(image_filepaths)
        for count, image_filepath in enumerate(image_filepaths):
            if (count+1) % 10 == 0:
                print("\r{0}/{1}".format(count+1, total), end='') 
            try:
                image, bounding_box = self.get_converted_sample(image_filepath, labels_dir)
                x_train.append(image)
                y_train.append(bounding_box)
            except IOError as ex:
                print("Couldn't open file: {0}".format(ex.filename))
                continue

        return (np.array(x_train), np.array(y_train))

    # returns image as numpy array, and labels as bounding box
    def get_converted_sample(self, image_filepath, labels_dir):
        image = self.get_normalised_image(image_filepath)
        image = cv2.resize(image, self.output_size)
        filename = os.path.basename(image_filepath).split('.')[0]
        label_filepath = os.path.join(labels_dir, "{0}.txt".format(filename))
        bounding_box = self.get_label(label_filepath)
        return (image, bounding_box)

    def get_label(self, filepath):
        with open(filepath, "r") as file:
            for line in file.readlines():
                x_centre, y_centre, width, height = map(float, line.strip().split(' '))
                label = (x_centre, y_centre, width, height)
                return label

    def get_normalised_image(self, filepath):
        with Image.open(filepath, mode='r') as img:
            pixel_data = np.asarray(img)[:,:,:3] / 255 # normalise
        return pixel_data