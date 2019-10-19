
from timeit import default_timer
from src.util import *

class Predictor:
    def __init__(self, model, input_size):
        self.model = model
        self.input_size = input_size

        self.last_bounding_box = (0, 0, 0, 0)

        self.last_time = 0
        self.acceleration = 2.5 
        self.additional_delay = 0

    def predict(self, image):
        curr_time = default_timer()
        dt = curr_time-self.last_time
        self.last_time = curr_time

        bounding_box = predict_bounding_box(self.model, image, size=self.input_size[:2][::-1])
        end = default_timer()
        
        x, y, width, height = bounding_box
        last_x, last_y, _, _ = self.last_bounding_box
        # calculate velocity
        delay = end - curr_time + self.additional_delay
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

        self.last_bounding_box = bounding_box
        return (bounding_box, real_bounding_box)