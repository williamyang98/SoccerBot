from pynput.keyboard import *

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