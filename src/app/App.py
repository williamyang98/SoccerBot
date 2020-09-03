from pynput.keyboard import *
from timeit import default_timer
import cv2
import numpy as np
import pyautogui
import threading
import time

from src.util import *
from src.app import *

class App:
    def __init__(self):
        # pyautogui flags
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = False

        self.is_running = False
        self.is_paused = True
        self.show_debug = False
        self.track_only = False

        self.bounding_box = (0, 0, 0, 0)
        self.image_callback = None
    
    def start(self, predictor, screenshotter):
        print("[Bot ready]")
        self.is_running = True
        self.start_prediction_thread(predictor, screenshotter)

    def start_prediction_thread(self, predictor, screenshotter):
        screen_x, screen_y, screen_width, screen_height = self.bounding_box

        last_y = 0
        last_frame_time = 0
        click_delay = 4e-3
        callback_delay = 0

        cooldown_duration = 0
        cooldown = 0

        while self.is_running:
            if not self.image_callback and self.is_paused:
                continue

            start = default_timer()
            image = screenshotter.get_screen_shot(self.bounding_box) 
            end = default_timer()
            screen_shot_delay = end-start

            predictor.additional_delay = screen_shot_delay+click_delay+callback_delay

            start = default_timer()

            prediction = predictor.predict(image)

            if prediction:
                detected_bounding_box, real_bounding_box = prediction
                x, y, _, _ = map_bounding_box(real_bounding_box, image.shape[:2])

            end = default_timer()
            model_time = end-start

            if self.show_debug:
                current_frame_time = default_timer()
                overall_time = current_frame_time-last_frame_time
                last_frame_time = current_frame_time
                print("\r{:.02f}ms/frame @ {:.02f}fps".format(overall_time*1000, 1/overall_time), end='')

            if prediction:
                reached_top = y < (screen_height * 0.50)

                x = x + screen_x
                y = y + screen_y

                dy = y-last_y
                last_y = y

                if self.check_mouse_inside(self.bounding_box, (x, y)) and not self.is_paused:
                    # if dy >= 0 and not reached_top or dy <= -70:
                    if (dy >= 0 and not reached_top) or dy >= 100:
                    # if (dy >= 0 and not reached_top):
                    # if dy >= 0 and not reached_top:
                    # if dy >= 0:
                    # if True:
                        if cooldown > 0:
                            cooldown -= 1
                        elif self.track_only:
                            pyautogui.moveTo(x=x, y=y)
                        else:
                            pyautogui.click(x=x, y=y)
                            cooldown = cooldown_duration
                    # else:
                        # pyautogui.moveTo(x=x, y=y)
            
            if self.image_callback is not None:
                start = default_timer()
                if prediction:
                    self.image_callback(image, detected_bounding_box, real_bounding_box)
                else:
                    self.image_callback(image, None, None)
                end = default_timer()    
                callback_delay = end-start

    def stop(self):
        self.is_running = False

    def check_mouse_inside(self, screen_bounding_box, pos):
        screen_x, screen_y, screen_width, screen_height = screen_bounding_box
        x, y = pos
        if x <= screen_x or x >= screen_x+screen_width:
            return False
        if y <= screen_y or y >= screen_y+screen_height:
            return False
        return True

    