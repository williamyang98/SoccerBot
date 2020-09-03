import glob
import argparse
import os

from src.generator import GeneratorConfig, BasicSampleGenerator
from create_tf_record_example import serialize_example
import cv2
import numpy as np
import tensorflow as tf

ASSETS_PATH = "assets/"
ICONS_PATH = os.path.join(ASSETS_PATH, "icons")
RECORDS_PATH = os.path.join(ASSETS_PATH, "data", "records")

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-size", default=10000, type=int)
    parser.add_argument("--total-records", default=1, type=int)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--icons-dir", default=ICONS_PATH)
    parser.add_argument("--records-dir", default=RECORDS_PATH)

    args = parser.parse_args()

    generator_config = get_generator_config(args.icons_dir)
    generator = BasicSampleGenerator(generator_config)

    os.makedirs(args.records_dir, exist_ok=True)

    create_records(args.records_dir, args.record_size, args.total_records, 
                   generator, args.override)


def get_generator_config(icons_dir):
    config = GeneratorConfig()
    config.set_background_image(os.path.join(icons_dir, "blank.png"))
    config.set_ball_image(os.path.join(icons_dir, "ball.png"))
    # get emotes
    emote_filepaths = glob.glob(os.path.join(icons_dir, "success*.png"))
    emote_filepaths.extend(glob.glob(os.path.join(icons_dir, "emote*.png")))
    config.set_emote_images(emote_filepaths)
    # score font
    config.set_score_font(os.path.join(ASSETS_PATH, "fonts", "segoeuil.ttf"), 92)

    return config

def create_records(directory, record_size, total_records, generator, override):
    idx = 0
    for i in range(total_records):
        print(f'Writing record {i:3d}/{total_records}')
        filename = f"images-{idx}-{record_size}.tfrec"
        filepath = os.path.join(directory, filename)

        while os.path.exists(filepath) and not override:
            idx += 1
            filename = f"images-{idx}-{record_size}.tfrec"
            filepath = os.path.join(directory, filename)

        with tf.io.TFRecordWriter(filepath) as writer:
            for i in range(record_size):
                image, (x_center, y_center, _, _), confidence = generator.create_sample()
                image = image.convert("RGB")
                image = np.array(image)
                image = cv2.imencode('.jpg', image, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
                
                example = serialize_example(
                    image, 
                    str.encode(f"IMAGE_{i}.jpg"),
                    x_center,
                    y_center,
                    confidence)
                writer.write(example)
                if i % 10 == 0: 
                    print(f'image {i:5d}/{record_size}\r', end='')

        idx += 1

if __name__ == '__main__':
    main()