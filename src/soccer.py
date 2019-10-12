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
delay = 0.1 

def main():
    lis = Listener(on_press=on_press)
    lis.start()

    rect = {'left': 677, 'top': 289, 'width': 325, 'height': 500}

    display_controls()


    with open("../assets/model/quantized-model.tflite", "rb") as file:
        model = LiteModel(file.read())

    last_falling_time = default_timer()
    consecutive_up_samples = 0

    lastX = rect['left'] 
    lastY = rect['top']
    lastTime = 0
    lastDelay = 0
    while running:
        start = default_timer()
        centreX, centreY = find_ball(model, rect)
        centreX = centreX + rect['left']
        centreY = centreY + rect['top'] 

        deltaX = centreX-lastX
        deltaY = centreY-lastY

        current_time = default_timer()
        delay = current_time-start

        delta_time = lastDelay + (start-lastTime)
        velocity_x = deltaX/delta_time
        velocity_y = deltaY/delta_time

        lastX = centreX
        lastY = centreY
        lastTime = current_time
        lastDelay = delay

        centreX = int(centreX + velocity_x*delay)
        centreY = int(centreY + velocity_y*delay)

        pyautogui.moveTo(x=centreX, y=centreY)
        if deltaY <= 0:
            consecutive_up_samples += 1
            if consecutive_up_samples > 5:
                consecutive_up_samples = 0
                last_falling_time = default_timer()
        else:
            consecutive_up_samples = 0

        if current_time-last_falling_time > delay and deltaY > 0 and check_mouse_inside(rect, (centreX, centreY)):
            pyautogui.click(x=centreX, y=centreY+40)

           
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