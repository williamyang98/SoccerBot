import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re

from src.util import draw_bounding_box

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")

    args = parser.parse_args()

    SCREEN_WIDTH = 322
    SCREEN_HEIGHT = 450

    import tensorflow as tf
    from load_tf_records import get_test_dataset

    dataset = get_test_dataset(
        [args.filename], 
        (SCREEN_HEIGHT, SCREEN_WIDTH))

    BATCH_SIZE = 100
    dataset = dataset.batch(BATCH_SIZE)

    for images, labels in dataset.take(100):
        for image, label in zip(images, labels):
            x, y, confidence = label

            image = np.array(image*255, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if confidence > 0.5:
                image = draw_bounding_box(image, (x, y, 0.27, 0.18), colour=(255,0,0))

            title = f"{x:.3f} {y:.3f} {confidence:.3f}"
            cv2.imshow("Preview", image)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

            # plt.title()
            # plt.imshow(image)
            # plt.show()

if __name__ == '__main__':
    main()