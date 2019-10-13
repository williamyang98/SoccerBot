import glob
import argparse
import os

from src.generator import GeneratorConfig, BasicSampleGenerator

ASSETS_PATH = "assets/"
ICONS_PATH = os.path.join(ASSETS_PATH, "icons")
IMAGES_OUTPUT_PATH = os.path.join(ASSETS_PATH, "samples")
LABELS_OUTPUT_PATH = os.path.join(ASSETS_PATH, "labels")

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-samples", type=int, default=100)
    parser.add_argument("--icons-dir", default=ICONS_PATH)
    parser.add_argument("--images-dir", default=IMAGES_OUTPUT_PATH)
    parser.add_argument("--labels-dir", default=LABELS_OUTPUT_PATH)
    parser.add_argument("--image-ext", default="JPEG")

    args = parser.parse_args()

    seed_folder(args.images_dir)
    seed_folder(args.labels_dir)

    generator_config = get_generator_config(args.icons_dir)
    generator = BasicSampleGenerator(generator_config)

    total_samples = args.total_samples
    for i in range(0, total_samples):
        if (i+1) % 10 == 0:
            print('\r{0}/{1}'.format(i+1, total_samples), end='')
        filename = "sample_{0}".format(i)
        image, bounding_box = generator.create_sample()

        image_filepath = os.path.join(args.images_dir, "{0}.{1}".format(filename, args.image_ext))
        save_image(image, image_filepath)

        label_filepath = os.path.join(args.labels_dir, "{0}.txt".format(filename))
        save_label(bounding_box, label_filepath)

def save_image(image, filepath, size=(256,256)):
    image = image.resize(size)
    image.save(filepath)

def save_label(bounding_box, filepath):
    x_centre, y_centre, width, height = bounding_box
    with open(filepath, "w") as file:
        file.write("0 {0} {1} {2} {3}".format(x_centre, y_centre, width, height))

def get_generator_config(icons_dir):
    config = GeneratorConfig()
    config.set_background_image(os.path.join(icons_dir, "blank.bmp"))
    config.set_ball_image(os.path.join(icons_dir, "ball.png"))
    # get emotes
    emote_filepaths = glob.glob(os.path.join(icons_dir, "success*.png"))
    emote_filepaths.extend(glob.glob(os.path.join(icons_dir, "emote*.png")))
    config.set_emote_images(emote_filepaths)
    # score font
    config.set_score_font(os.path.join(ASSETS_PATH, "fonts", "segoeuil.ttf"), 92)

    return config

def seed_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    

if __name__ == '__main__':
    main()