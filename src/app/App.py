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
    def __init__(self, show_debug=False, show_preview=False):
        # pyautogui flags
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = False

        self.is_running = False
        self.is_paused = True
        self.show_debug = show_debug
        self.show_preview = show_preview
        self.track_only = False
        self.recording = False
        self.bounding_box = (0, 0, 0, 0)
        
        self.resume_key = Key.f1
        self.pause_key = Key.f2
        self.exit_key = Key.f3
        self.preview_key = Key.f4
        self.debug_key = Key.f5
        self.track_key = Key.f6
        self.record_key = Key.f7

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
        click_delay = 4e-3
        preview_delay = 0

        predictor.acceleration = 2.5

        cooldown_duration = 0
        cooldown = 0

        while self.is_running:
            if not self.show_preview and not self.recording and self.is_paused:
                continue

            start = default_timer()
            image = screenshotter.get_screen_shot(self.bounding_box) 
            end = default_timer()
            screen_shot_delay = end-start

            predictor.additional_delay = screen_shot_delay+click_delay+preview_delay

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
            
            if self.show_preview or self.recording:
                start = default_timer()
                with self.preview_thread_lock:
                    if prediction:
                        self.preview = (image, detected_bounding_box, real_bounding_box)
                    else:
                        self.preview = (image, None, None)
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
        print("F6 : Track only")
        print("F7 : Record")
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
        elif key == self.track_key:
            self.track_only = not self.track_only
            print("[Track={0}]".format(self.track_only))
        elif key == self.record_key:
            self.recording = not self.recording
            print(f"[Recording={self.recording}]")

    def check_mouse_inside(self, screen_bounding_box, pos):
        screen_x, screen_y, screen_width, screen_height = screen_bounding_box
        x, y = pos
        if x <= screen_x or x >= screen_x+screen_width:
            return False
        if y <= screen_y or y >= screen_y+screen_height:
            return False
        return True

    def start_preview_thread(self):
        window_name = "Preview"
        _, _, width, height = self.bounding_box
        video_out = cv2.VideoWriter(
            "video.mp4",
            cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
        total_frames = 0

        while self.is_running:
            if not self.show_preview:
                cv2.destroyWindow(window_name)
                time.sleep(30e-3)

            if not self.show_preview and not self.recording:
                continue

            with self.preview_thread_lock:
                preview = self.preview
                self.preview = None

            if preview is None:
                continue
            # construct preview 
            image, detected_bounding_box, real_bounding_box = preview 

            if detected_bounding_box is not None:
                draw_bounding_box(image, detected_bounding_box)
                draw_bounding_box(image, real_bounding_box, (255, 0, 0))

            # display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.show_preview:
                cv2.imshow(window_name, image)

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
            
            if self.recording:
                video_out.write(image)
                total_frames += 1
        
        video_out.release()
        print(f"Wrote {total_frames} frames")