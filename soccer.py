import pyautogui
from pynput.keyboard import *
import numpy as np
import cv2
from timeit import default_timer

from src.util import *
from src.app import *

import argparse

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

INPUT_SHAPE = (160,227)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action='store_true')
    parser.add_argument("--model", default="assets/model/model.h5")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # screen box is (x, y, width, height)
    screen_bounding_box = (677, 289, 325, 500)
    screen_x, screen_y, screen_height, screen_width = screen_bounding_box

    app = App()
    app.start()

    from src.model import LiteModel, Model

    if args.lite:
        with open(args.model, "rb") as file:
            model = LiteModel(file.read())
    else:
        model = Model.load(args.model)

    predictor = Predictor(model)
    predictor.acceleration = 5 

    # screenshotter = D3DScreenshot(screen_bounding_box)
    screenshotter = MSSScreenshot(screen_bounding_box)

    last_y = 0
    click_delay = 0.020

    print("[Bot ready]")
    while app.is_running:
        if not args.preview and app.is_paused:
            continue

        start = default_timer()
        image = screenshotter.get_screen_shot() 
        end = default_timer()
        screen_shot_delay = end-start

        predictor.additional_delay = screen_shot_delay+click_delay

        detected_bounding_box, real_bounding_box = predictor.predict(image)
        x, y, _, _ = map_bounding_box(real_bounding_box, image.shape[:2])

        if args.debug:
            net_dt = default_timer()-start
            print("\r{:.02f}ms/frame".format(net_dt*1000), end='')

        reached_top = y < screen_height * 0.30

        x = x + screen_x
        y = y + screen_y

        dy = y-last_y
        last_y = y

        if check_mouse_inside(screen_bounding_box, (x, y)) and not app.is_paused:
            # if dy >= 0 and not reached_top:
            if dy >= 0:
                pyautogui.click(x=x, y=y)

        if args.preview:
            draw_bounding_box(image, detected_bounding_box)
            draw_bounding_box(image, real_bounding_box, (255, 0, 0))
            show_preview(image)

def check_mouse_inside(screen_bounding_box, pos):
    screen_x, screen_y, screen_height, screen_width = screen_bounding_box
    x, y = pos
    if x <= screen_x or x >= screen_x+screen_width:
        return False
    if y <= screen_y or y >= screen_y+screen_height:
        return False
    return True

def show_preview(preview):
    cv2.imshow("Preview", preview)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
