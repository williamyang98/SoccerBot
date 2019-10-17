import glob
import argparse
import os

from src.generator import GeneratorConfig, BasicSampleGenerator

ASSETS_PATH = "assets/"
ICONS_PATH = os.path.join(ASSETS_PATH, "icons")
IMAGES_OUTPUT_PATH = os.path.join(ASSETS_PATH, "data", "samples")
LABELS_OUTPUT_PATH = os.path.join(ASSETS_PATH, "data", "labels.txt")

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-samples", type=int, default=100)
    parser.add_argument("--icons-dir", default=ICONS_PATH)
    parser.add_argument("--images-dir", default=IMAGES_OUTPUT_PATH)
    parser.add_argument("--labels-file", default=LABELS_OUTPUT_PATH)
    parser.add_argument("--image-ext", default="JPEG")

    args = parser.parse_args()

    seed_folder(args.images_dir)

    generator_config = get_generator_config(args.icons_dir)
    generator = BasicSampleGenerator(generator_config)

    with open(args.labels_file, "w+") as file:
        header, labels = generate_images(args.images_dir, args.total_samples, generator, args.image_ext)        
        file.write(" ".join(header)+"\n")
        for label in labels:
            file.write(" ".join(map(str, label))+"\n")



def generate_images(output_dir, total_samples, generator, extension):
    header = ["filename", "x_centre", "y_centre", "width", "height"]
    labels = []
    for i in range(0, total_samples):
        if (i+1) % 10 == 0:
            print('\r{0}/{1}'.format(i+1, total_samples), end='')
        filename = "sample_{0}.{1}".format(i, extension)
        image_filepath = os.path.join(output_dir, filename)

        image, bounding_box = generator.create_sample()
        save_image(image, image_filepath)

        labels.append((filename,)+bounding_box)
    return (header, labels)


def save_image(image, filepath, size=None):
    if size:
        image = image.resize(size)
    rgb_image = image.convert("RGB")
    rgb_image.save(filepath)

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

def seed_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    

if __name__ == '__main__':
    main()