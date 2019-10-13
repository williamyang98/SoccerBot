from PIL import Image
from timeit import default_timer
import argparse
import numpy as np
import os
import glob

from src.model import Model, LiteModel
from src.util import *

INPUT_SIZE = (256, 256, 3)

IMAGES_DIR = "assets/data/samples/"
PREDICTION_DIR = "assets/predictions/"
MODEL_DIR = "assets/model/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", default=os.path.join(IMAGES_DIR, "*JPEG"))
    parser.add_argument("--output-dir", default=os.path.join(PREDICTION_DIR))
    parser.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "model.h5"))
    parser.add_argument("--lite", action="store_true")

    args = parser.parse_args()

    # fetch appropriate model
    if not args.lite:
        print("Loading full model: {0}".format(args.model))
        model = Model(INPUT_SIZE, (4,))
        model.load(args.model)
    else:
        print("Loading quantized model: {0}".format(args.model))
        with open(args.model, 'rb') as file:
            model = LiteModel(file.read())

    # process all images
    filepaths = glob.glob(args.input_files)
    parse_images(model, filepaths, args.output_dir)

def parse_images(model, filepaths, output_dir):
    count = len(filepaths)
    times = []

    for i, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        if (i+1) % 10 == 0:
            print("\r{0}/{1} @ {2:.02f}ms/sample".format(i+1, count, 1000*sum(times)/len(times)), end="")
        try:
            output_img, elapsed_time = parse_image(model, filepath)
            Image.fromarray(output_img).save(os.path.join(output_dir, filename))
            times.append(elapsed_time)
        except Exception as ex:
            print("Couldn't process image: {0}".format(filepath))
            raise(ex)

    print()

def parse_image(model, input_file):
    image = Image.open(input_file)
    image = np.array(image)

    start = default_timer()
    bounding_box = predict_bounding_box(model, image)
    end = default_timer()

    draw_bounding_box(image, bounding_box)
    elapsed = end-start
    return (image, elapsed)

if __name__ == '__main__':
    main()




