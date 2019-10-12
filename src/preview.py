import numpy as np
import mss
import cv2
import time
import pyautogui

def main():
    rect = {'left': 677, 'top': 289, 'width': 325, 'height': 500}
    while(True):
        with mss.mss() as screen:
            image = screen.grab(rect)
        
        image = np.array(image)
        cv2.imshow('window', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()