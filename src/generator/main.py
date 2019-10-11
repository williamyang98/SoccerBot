import os
import argparse
from generator import create_sample, SAVE_PATH

def main():    
    parser = argparse.ArgumentParser()

    img_format = "png"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    total_samples = 10000
    for i in range(0, total_samples):
        if (i+1) % 10 == 0:
            print('\r{0}/{1}'.format(i+1, total_samples), end='')
        file = "sample_{0}.{1}".format(i, img_format)
        sample_img = create_sample()
        sample_img.save(os.path.join(SAVE_PATH, file), format=img_format)

if __name__ == '__main__':
    main()