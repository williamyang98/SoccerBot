import numpy as np
import mss
import cv2
import time
import pyautogui

from model.lite_model import LiteModel

def find_ball(model, image):
    x = image[:,:,:3] / 255
    x = cv2.resize(x, (256,256))
    # only get first 3 channels 
    y = model.predict(np.asarray([x]))[0] 
    
    return y

def draw_bounding_box(image, bounding_box):
    image_width, image_height, image_channels = image.shape
    x_centre, y_centre, width, height = bounding_box 

    left = int((x_centre-width/2)*image_width)
    right = int((x_centre+width/2)*image_width)
    top = int((y_centre-height/2)*image_height)
    bottom = int((y_centre+height/2)*image_height)

    cv2.rectangle(image, (left, top), (right, bottom), (0, 20, 200), 2)
    return image


def main():
    rect = {'left': 677, 'top': 289, 'width': 325, 'height': 500}

    with open("../assets/model/quantized-model.tflite", "rb") as file:
        model = LiteModel(file.read())

    while(True):
        with mss.mss() as screen:
            image = screen.grab(rect)
        image = np.array(image)
        bounding_box = find_ball(model, image)
        image = draw_bounding_box(cv2.resize(image, (500, 500)), bounding_box)
        cv2.imshow('window', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()