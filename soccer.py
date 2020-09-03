import argparse
import threading
import cv2
import time
import numpy as np
from pynput.keyboard import *

from src.app import App, Predictor, MSSScreenshot
from src.util import draw_bounding_box

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="assets/models/cnn_227_160_quantized.tflite")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()

    from load_model import load_any_model_filepath
    model, (HEIGHT, WIDTH) = load_any_model_filepath(args.model, not args.checkpoint, args.large)

    # screen box is (x, y, width, height)
    bounding_box = (677, 289, 322, 455)
    app = App()
    app.bounding_box = bounding_box

    callback = ImageCallback(*bounding_box[2:4])
    app.image_callback = callback.on_image

    screenshotter = MSSScreenshot()
    # screenshotter = D3DScreenshot()

    predictor = Predictor(model, (HEIGHT, WIDTH))
    predictor.acceleration = 2.5

    def on_press(key):
        if key == Key.f1:
            print("[Resumed]")
            app.is_paused = False
        elif key == Key.f2:
            print("[Paused]")
            app.is_paused = True
        elif key == Key.f3:
            print("[Exit]")
            app.stop()
            # callback.stop_threads()
        elif key == Key.f4:
            callback.is_preview = not callback.is_preview
            print(f"[Preview={callback.is_preview}]")
        elif key == Key.f5:
            app.show_debug = not app.show_debug
            print("[Debug={0}]".format(app.show_debug))
        elif key == Key.f6:
            app.track_only = not app.track_only
            print("[Track={0}]".format(app.track_only))
        elif key == Key.f7:
            callback.is_recording = not callback.is_recording
            print(f"[Recording={callback.is_recording}]")
    

    input_listener = Listener(on_press=on_press)
    input_listener.start()
    display_controls()

    callback.start_threads()
    app.start(predictor, screenshotter)
    input_listener.stop()
    callback.stop_threads()

def display_controls():
    print("== Controls ==")
    print("F1 : Resume")
    print("F2 : Pause")
    print("F3 : Exit")
    print("F4 : Preview key")
    print("F5 : Debug key")
    print("F6 : Track only")
    print("F7 : Record")
    print("==============")

class ImageCallback:
    def __init__(self, width, height):
        self.frame = 0
        self.width = width
        self.height = height

        self.image = None 
        self.is_running = False
        self.is_recording = False
        self.is_preview = False

        self.thread_record = threading.Thread(target=self.start_video_thread)
        self.thread_preview = threading.Thread(target=self.start_preview_thread)
    
    def start_threads(self):
        self.is_running = True
        self.thread_record.start()
        self.thread_preview.start()
    
    def stop_threads(self):
        self.is_running = False
        self.thread_record.join()
        self.thread_preview.join()

    def on_image(self, image, pred_bb, real_bb):
        if pred_bb is not None:
            image = draw_bounding_box(image, pred_bb)
            image = draw_bounding_box(image, real_bb, (255, 0, 0))
            # display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.frame += 1
    
    def start_video_thread(self):
        writer = cv2.VideoWriter(
            "video.mp4",
            cv2.VideoWriter_fourcc(*'DIVX'), 
                30, 
                (self.width, self.height))

        last_frame = None
        total_frames = 0

        while self.is_running:
            image = self.image
            if not self.is_recording or image is None:
                time.sleep(15e-3)
                continue

            if last_frame == self.frame:
                time.sleep(2e-3)
                continue
            
            last_frame = self.frame
            total_frames += 1

            writer.write(image)

        writer.release()
        print(f"Wrote {total_frames} frames")

    def start_preview_thread(self):
        window_name = "Preview"

        last_frame = None

        while self.is_running:
            if not self.is_preview:
                cv2.destroyWindow(window_name)
                time.sleep(15e-3)
                continue
        
            if last_frame == self.frame or self.image is None:
                time.sleep(15e-3)
                continue
            
            last_frame = self.frame
            cv2.imshow(window_name, self.image)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            
if __name__ == "__main__":
    main()
