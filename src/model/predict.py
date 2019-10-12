from PIL import Image, ImageDraw
from timeit import default_timer
import argparse
import numpy as np
import os
import glob

from model import Model
from lite_model import LiteModel
from paths import MODEL_DIR, ASSETS_DIR

INPUT_SIZE = (256, 256, 3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", default=os.path.join(ASSETS_DIR, "samples", "*JPEG"))
    parser.add_argument("--output-dir", default=os.path.join(ASSETS_DIR, "predictions"))
    parser.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "model.h5"))
    parser.add_argument("--lite", action="store_true")

    args = parser.parse_args()

    # fetch appropriate model
    if not args.lite:
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
            output_img.save(os.path.join(output_dir, filename))
            times.append(elapsed_time)
        except Exception as ex:
            print("Couldn't process image: {0}".format(filepath))
            raise(ex)

    print()

def parse_image(model, input_file):
    img = Image.open(input_file)
    X = convert_image(img)
    Y, elapsed_time = predict(model, X)
    draw_bounding_box(img, Y)
    return (img, elapsed_time)

def draw_bounding_box(img, bounding_box):
    width, height = img.size
    x_centre_norm, y_centre_norm, width_norm, height_norm = bounding_box

    left = (x_centre_norm-width_norm/2.0)*width
    right = (x_centre_norm+width_norm/2.0)*width
    top = (y_centre_norm-height_norm/2.0)*height
    bottom = (y_centre_norm+height_norm/2.0)*height

    rect = [(left, top), (right, bottom)]

    draw = ImageDraw.Draw(img)
    draw.rectangle(rect, outline=(255,0,0), width=2)

def predict(model, X):
    start = default_timer()
    Y = model.predict(np.asarray([X]))[0]
    end = default_timer()
    elapsed = end-start
    return (Y, elapsed)

def convert_image(img):
    img = img.resize((256,256))
    X = np.asarray(img) / 255
    return X

if __name__ == '__main__':
    main()




