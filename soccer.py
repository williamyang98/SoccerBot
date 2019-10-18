import pyautogui
from pynput.keyboard import *
import numpy as np
import mss
import cv2
from timeit import default_timer

from src.util import *

import argparse

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

INPUT_SHAPE = (160,227,3)

class App:
    def __init__(self):
        self.is_running = False
        self.is_paused = True
        
        self.resume_key = Key.f1
        self.pause_key = Key.f2
        self.exit_key = Key.f3

        self.listener = Listener(on_press=self.on_press)

    def on_press(self, key):
        if key == self.resume_key:
            print("\r[Resumed]")
            self.is_paused = False
        elif key == self.pause_key:
            print("\r[Paused]")
            self.is_paused = True
        elif key == self.exit_key:
            print("\r[Exit]")
            self.stop()

    def start(self):
        self.listener.start()
        self.is_running = True
        self.display_controls()

    def stop(self):
        self.is_running = False
        self.listener.stop()

    def display_controls(self):
        print("== Controls ==")
        print("F1 : Resume")
        print("F2 : Pause")
        print("F3 : Exit")
        print("==============")

class Predictor:
    def __init__(self, model):
        self.model = model
        self.last_bounding_box = (0, 0, 0, 0)
        self.last_time = 0
        self.acceleration = 2.5 # normalised
        self.screen_shot_delay = 0

    def predict(self, image):
        curr_time = default_timer()
        dt = curr_time-self.last_time
        self.last_time = curr_time

        bounding_box = predict_bounding_box(self.model, image, size=(INPUT_SHAPE[1], INPUT_SHAPE[0]))
        end = default_timer()
        
        x, y, width, height = bounding_box
        last_x, last_y, _, _ = self.last_bounding_box
        # calculate velocity
        delay = end - curr_time + self.screen_shot_delay
        vx, vy = (x-last_x)/dt, (y-last_y)/dt
        real_x = x +  vx*delay
        real_y = y + vy*delay 
        if vy != 0 and vx != 0:
            real_y += self.acceleration*0.5*(delay**2) # when ball is stationary

        # calculate bounce
        right_border = 1-width/2
        left_border = width/2
        if real_x > right_border:
            delta = real_x-right_border
            real_x = right_border-delta
        elif real_x < left_border:
            delta = left_border-real_x
            real_x = left_border+delta

        real_bounding_box = (real_x, real_y, width, height)

        self.last_bounding_box = bounding_box
        return (bounding_box, real_bounding_box)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action='store_true')
    parser.add_argument("--model", default="assets/model/model.h5")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    rect = {'left': 677, 'top': 289, 'width': 325, 'height': 500}

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

    if args.debug:
        screen_shot_delay = get_screen_shot_delay(rect)
        print("Screen shot delay: {:.02f}ms".format(screen_shot_delay*1000)) 
        predictor.screen_shot_delay = screen_shot_delay


    last_y = 0
    click_delay = 0.020

    print("[Bot ready]")
    while app.is_running:
        if not args.preview and app.is_paused:
            continue

        start = default_timer()
        with mss.mss() as screen:
            image = screen.grab(rect)

        image = np.array(image)
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        predictor.screen_shot_delay = default_timer()-start + click_delay

        detected_bounding_box, real_bounding_box = predictor.predict(converted_image)
        x, y, _, _ = map_bounding_box(real_bounding_box, image.shape[:2])

        if args.debug:
            net_dt = default_timer()-start
            print("\r{:.02f}ms/frame".format(net_dt*1000), end='')

        reached_top = y < rect['height'] * 0.30

        x = x + rect['left']
        y = y + rect['top'] 

        dy = y-last_y
        last_y = y

        if check_mouse_inside(rect, (x, y)) and not app.is_paused:
            # if dy >= 0 and not reached_top:
            if dy >= 0:
                pyautogui.click(x=x, y=y)

        if args.preview:
            draw_bounding_box(image, detected_bounding_box)
            draw_bounding_box(image, real_bounding_box, (255, 0, 0))
            show_preview(image)


def get_screen_shot_delay(rect):
    start = default_timer()
    with mss.mss() as screen:
        screen.grab(rect)
    end = default_timer()
    screen_shot_delay = end-start
    return screen_shot_delay


def check_mouse_inside(rect, pos):
    x, y = pos
    if x <= rect['left'] or x >= rect['left']+rect['width']:
        return False
    if y <= rect['top'] or y >= rect['top']+rect['height']:
        return False
    return True


def show_preview(preview):
    cv2.imshow("Preview", preview)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
