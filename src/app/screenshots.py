import d3dshot
import mss
import cv2
import numpy as np

class D3DScreenshot:
    def __init__(self, bounding_box):
        left, top, width, height = bounding_box
        self.region = (left, top, left+width, top+height)
        self.d3dshotter = d3dshot.create(capture_output='numpy')

    def get_screen_shot(self):
        image = self.d3dshotter.screenshot(region=self.region)
        return image

class MSSScreenshot:
    def __init__(self, bounding_box):
        left, top, width, height = bounding_box
        self.monitor = {'left': left, 'top': top, 'width': width, 'height': height}
        self.screen = mss.mss()
    
    def get_screen_shot(self):
        image = self.screen.grab(self.monitor)
        image = np.array(image)
        # convert from bgra to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image