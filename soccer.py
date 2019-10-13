import pyautogui
from pynput.keyboard import *
import numpy as np
import mss
import cv2
from timeit import default_timer

from src.model.lite_model import LiteModel
from src.util import *

import argparse

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
#  autoclick buttons
resume_key = Key.f1
pause_key = Key.f2
exit_key = Key.f3

pause = True
running = True

def on_press(key):
    global running, pause
    if key == resume_key:
        pause = False
        print("[Resumed]")
    elif key == pause_key:
        pause = True
        print("[Paused]")
    elif key == exit_key:
        running = False
        print("[Exit]")
        quit()

def display_controls():
    print("\t F1 = Resume")
    print("\t F2 = Pause")
    print("\t F3 = Exit")

# Number of pixels under ball center to click
delay = 0.1 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action='store_true')

    args = parser.parse_args()

    lis = Listener(on_press=on_press)
    lis.start()

    rect = {'left': 677, 'top': 289, 'width': 325, 'height': 500}

    display_controls()

    with open("assets/model/quantized-model.tflite", "rb") as file:
        model = LiteModel(file.read())

    while running:
        if not args.preview and pause:
            continue
        
        start = default_timer()
        with mss.mss() as screen:
            image = screen.grab(rect)
        image = np.array(image)
        bounding_box = predict_bounding_box(model, image)

        centreX, centreY, _, _ = map_bounding_box(bounding_box, image.shape[:2])
        centreX = centreX + rect['left']
        centreY = centreY + rect['top'] 
        end = default_timer()
        print("\r{:.02f}ms/frame".format((end-start)*1000), end='')

        if check_mouse_inside(rect, (centreX, centreY)) and not pause:
            pyautogui.moveTo(x=centreX, y=centreY)
            pyautogui.click(x=centreX, y=centreY)
        
        if args.preview:
            preview = draw_bounding_box(image, bounding_box)
            cv2.imshow("Preview", preview)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    lis.stop()

def check_mouse_inside(rect, pos):
    x, y = pos
    if x <= rect['left'] or x >= rect['left']+rect['width']:
        return False
    if y <= rect['top'] or y >= rect['top']+rect['height']:
        return False
    return True


if __name__ == "__main__":
    main()