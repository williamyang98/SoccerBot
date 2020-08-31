from PIL import Image
from timeit import default_timer
import argparse
import numpy as np
import os
import glob

from src.util import *

import re

IMAGES_DIR = "assets/data/samples/"
PREDICTION_DIR = "assets/data/predictions/"
MODEL_DIR = "assets/models/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", default=os.path.join(IMAGES_DIR, "*JPEG"))
    parser.add_argument("--output-dir", default=os.path.join(PREDICTION_DIR))
    parser.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "cnn_113_80.h5f"))
    parser.add_argument("--large", action="store_true")

    args = parser.parse_args()

    from load_model import load_from_filepath
    model, (HEIGHT, WIDTH) = load_from_filepath(args.model, args.large) 

    # process all images
    filepaths = glob.glob(args.input_files)
    parse_images(model, filepaths, args.output_dir, (HEIGHT, WIDTH))

def parse_images(model, filepaths, output_dir, input_size):
    count = len(filepaths)
    times = []

    for i, filepath in enumerate(sorted(filepaths)):
        filename = os.path.basename(filepath)
        if (i+1) % 10 == 0:
            print("\r{0}/{1} @ {2:.02f}ms/sample".format(i+1, count, 1000*sum(times)/len(times)), end="")
        try:
            output_img, elapsed_time = parse_image(model, filepath, input_size)
            Image.fromarray(output_img).save(os.path.join(output_dir, filename))
            times.append(elapsed_time)
        except Exception as ex:
            print("Couldn't process image: {0}".format(filepath))
            raise(ex)

    print()

def parse_image(model, input_file, input_size):
    image = Image.open(input_file)
    image = np.array(image)

    start = default_timer()

    np_img = image[:,:,:3] / 255
    np_img = cv2.resize(np_img, input_size[::-1], interpolation=cv2.INTER_NEAREST)
    x_pos, y_pos, confidence = model.predict(np.asarray([np_img]))[0]
    end = default_timer()

    bounding_box = (x_pos, y_pos, 0.27, 0.18)
    if confidence > 0.8:
        draw_bounding_box(image, bounding_box)

    elapsed = end-start
    return (image, elapsed)

if __name__ == '__main__':
    main()




