import pyautogui
from pynput.keyboard import *
import numpy as np
import mss
import cv2
from timeit import default_timer
from PIL import Image

from model.lite_model import LiteModel


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# take sc, process and get coord
def find_ball(model, rect):
    with mss.mss() as screen:
        image = screen.grab(rect)

    data = np.array(image)
    x = data[:,:,:3] / 255
    x = cv2.resize(x, (256,256))
    # only get first 3 channels 
    y = model.predict(np.asarray([x]))[0] 
    
    x_centre, y_centre, width, height = y

    width = rect['width']
    height = rect['height']
    
    return x_centre*width, y_centre*height
    

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
    print("// AutoClicker by NeedtobeatVictor")
    print("// - Settings: ")
    print("\t delay = " + str(delay) + ' sec' + '\n')
    print("// - Controls:")
    print("\t F1 = Resume")
    print("\t F2 = Pause")
    print("\t F3 = Exit")
    print("\t ESC = For real Exit")
    print("-----------------------------------------------------")
    print('Don\'t need to Press F1 to start ...')

# Number of pixels under ball center to click
delay = 0.2 

def main():
    lis = Listener(on_press=on_press)
    lis.start()

    rect = {'left': 677, 'top': 289, 'width': 325, 'height': 500}

    display_controls()


    with open("../assets/model/quantized-model.tflite", "rb") as file:
        model = LiteModel(file.read())

    last_click = default_timer()
    while running:
        centreX, centreY = find_ball(model, rect)
        centreX = centreX + rect['left']
        centreY = centreY + rect['top'] 
        current_time = default_timer()
        pyautogui.moveTo(x=centreX, y=centreY)
        if current_time-last_click > delay:
            last_click = current_time
            pyautogui.click(x=centreX, y=centreY)

           
    lis.stop()


if __name__ == "__main__":
    main()