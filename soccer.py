from pynput.keyboard import *
from timeit import default_timer
import argparse
import cv2
import numpy as np
import pyautogui
import threading
import time

from src.util import *
from src.app import *

# pyautogui flags
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

INPUT_SHAPE = (160, 227)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action='store_true')
    parser.add_argument("--model", default="assets/model/model.h5")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    

    from src.model import LiteModel, Model

    if args.lite:
        with open(args.model, "rb") as file:
            model = LiteModel(file.read())
    else:
        model = Model.load(args.model)

    # screen box is (x, y, width, height)
    app = App(args.debug, args.preview)
    app.bounding_box = (677, 289, 325, 500)

    predictor = Predictor(model, INPUT_SHAPE)
    predictor.acceleration = 5

    # screenshotter = D3DScreenshot()
    screenshotter = MSSScreenshot()

    app.start(predictor, screenshotter)

class App:
    def __init__(self, show_debug=False, show_preview=False):
        self.is_running = False
        self.is_paused = True
        self.show_debug = show_debug
        self.show_preview = show_preview
        self.bounding_box = (0, 0, 0, 0)
        
        self.resume_key = Key.f1
        self.pause_key = Key.f2
        self.exit_key = Key.f3
        self.preview_key = Key.f4
        self.debug_key = Key.f5

        self.input_listener = Listener(on_press=self.on_press)
        self.preview = None
        self.preview_thread_lock = threading.RLock()
    
    def start(self, predictor, screenshotter):
        print("[Bot ready]")
        self.input_listener.start()
        self.is_running = True
        self.display_controls()

        preview_thread = threading.Thread(target=self.start_preview_thread)

        preview_thread.start()
        self.start_prediction_thread(predictor, screenshotter)

        preview_thread.join()

    def start_prediction_thread(self, predictor, screenshotter):
        screen_x, screen_y, screen_width, screen_height = self.bounding_box

        last_y = 0
        last_frame_time = 0
        click_delay = 0.020
        preview_delay = 0

        while self.is_running:
            if not self.show_preview and self.is_paused:
                continue

            start = default_timer()
            image = screenshotter.get_screen_shot(self.bounding_box) 
            end = default_timer()
            screen_shot_delay = end-start

            predictor.additional_delay = screen_shot_delay+click_delay+preview_delay

            start = default_timer()
            detected_bounding_box, real_bounding_box = predictor.predict(image)
            x, y, _, _ = map_bounding_box(real_bounding_box, image.shape[:2])
            end = default_timer()
            model_time = end-start

            if self.show_debug:
                current_frame_time = default_timer()
                overall_time = current_frame_time-last_frame_time
                last_frame_time = current_frame_time
                print("\r{:.02f}ms/frame @ {:.02f}fps".format(overall_time*1000, 1/overall_time), end='')

            reached_top = y < (screen_height * 0.30)

            x = x + screen_x
            y = y + screen_y

            dy = y-last_y
            last_y = y

            if self.check_mouse_inside(self.bounding_box, (x, y)) and not self.is_paused:
                # if dy >= 0 and not reached_top:
                if dy >= 0:
                    pyautogui.click(x=x, y=y)
            
            if self.show_preview:
                start = default_timer()
                with self.preview_thread_lock:
                    self.preview = (image, detected_bounding_box, real_bounding_box)
                end = default_timer()    
                preview_delay = end-start


    def stop(self):
        self.is_running = False
        self.input_listener.stop()

    def display_controls(self):
        print("== Controls ==")
        print("F1 : Resume")
        print("F2 : Pause")
        print("F3 : Exit")
        print("F4 : Preview key")
        print("F5 : Debug key")
        print("==============")

    def on_press(self, key):
        if key == self.resume_key:
            print("[Resumed]")
            self.is_paused = False
        elif key == self.pause_key:
            print("[Paused]")
            self.is_paused = True
        elif key == self.exit_key:
            print("[Exit]")
            self.stop()
        elif key == self.preview_key:
            self.show_preview = not self.show_preview
            print("[Preview={0}]".format(self.show_preview))
        elif key == self.debug_key:
            self.show_debug = not self.show_debug
            print("[Debug={0}]".format(self.show_debug))

    def check_mouse_inside(self, screen_bounding_box, pos):
        screen_x, screen_y, screen_width, screen_height = screen_bounding_box
        x, y = pos
        if x <= screen_x or x >= screen_x+screen_width:
            return False
        if y <= screen_y or y >= screen_y+screen_height:
            return False
        return True

    def start_preview_thread(self):
        while self.is_running:
            if not self.show_preview:
                time.sleep(0.03)
                continue

            with self.preview_thread_lock:
                preview = self.preview
                self.preview = None

            if preview is None:
                continue
            # construct preview 
            image, detected_bounding_box, real_bounding_box = preview 
            draw_bounding_box(image, detected_bounding_box)
            draw_bounding_box(image, real_bounding_box, (255, 0, 0))
            # display
            cv2.imshow("Preview", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(100) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
