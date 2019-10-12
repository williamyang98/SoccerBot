import os
import sys
import argparse
from generator import create_sample
from paths import DATA_PATH, LABELS_PATH

IMAGE_SIZE = (256, 256)

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-samples", type=int, default=100)

    args = parser.parse_args()


    seed_folder(DATA_PATH)
    seed_folder(LABELS_PATH)

    total_samples = args.total_samples
    for i in range(0, total_samples):
        if (i+1) % 10 == 0:
            print('\r{0}/{1}'.format(i+1, total_samples), end='')
        filename = "sample_{0}".format(i)
        sample = create_sample()
        save_sample(sample, filename)

def save_sample(sample, filename, img_format="JPEG"):
    img, (x_centre, y_centre, width, height) = sample
    img = img.resize(IMAGE_SIZE)
    # create image
    img_name = "{0}.{1}".format(filename, img_format)
    img.save(os.path.join(DATA_PATH, img_name), format=img_format)
    # create labels
    labels_name = "{0}.txt".format(filename)
    with open(os.path.join(LABELS_PATH, labels_name), "w") as file:
        file.write("0 {0} {1} {2} {3}".format(x_centre, y_centre, width, height))

def seed_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    

if __name__ == '__main__':
    main()