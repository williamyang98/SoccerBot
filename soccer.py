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
        print("\t F1 = Resume")
        print("\t F2 = Pause")
        print("\t F3 = Exit")

class Predictor:
    def __init__(self, model):
        self.model = model
        self.last_bounding_box = (0, 0, 0, 0)
        self.last_time = 0
        self.acceleration = 2.5 # normalised
        self.screen_shot_delay = 0

    def predict(self, image, show_preview=False):
        curr_time = default_timer()
        dt = curr_time-self.last_time
        self.last_time = curr_time

        bounding_box = predict_bounding_box(self.model, image)
        end = default_timer()
        print("\r{:.02f}ms/frame".format(dt*1000), end='')
        
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

        if show_preview:
            image = draw_bounding_box(image, bounding_box)
            image = draw_bounding_box(image, real_bounding_box, (255, 0, 0))
            self.show_preview(image)
        
        self.last_bounding_box = bounding_box
        return real_bounding_box

    
    def show_preview(self, preview):
        cv2.imshow("Preview", preview)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action='store_true')
    parser.add_argument("--model", default="assets/model/quantized-model.tflite")

    args = parser.parse_args()

    rect = {'left': 677, 'top': 289, 'width': 325, 'height': 500}

    app = App()
    app.start()

    with open(args.model, "rb") as file:
        model = LiteModel(file.read())
    
    predictor = Predictor(model)
    predictor.acceleration = 8

    screen_shot_delay = get_screen_shot_delay(rect)
    print("Screen shot delay: {:.02f}ms".format(screen_shot_delay*1000)) 
    predictor.screen_shot_delay = screen_shot_delay

    while app.is_running:
        if not args.preview and app.is_paused:
            continue
        
        start = default_timer()
        with mss.mss() as screen:
            image = screen.grab(rect)

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        bounding_box = predictor.predict(image, show_preview=args.preview)

        x, y, _, _ = map_bounding_box(bounding_box, image.shape[:2])

        reached_top = y < 150

        x = x + rect['left']
        y = y + rect['top'] 

        if check_mouse_inside(rect, (x, y)) and not app.is_paused and not reached_top:
            pyautogui.moveTo(x=x, y=y)
            pyautogui.click(x=x, y=y)

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


if __name__ == "__main__":
    main()