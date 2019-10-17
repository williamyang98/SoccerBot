import numpy as np
import math
import random

from .overlays import *

GREY = (100, 100, 100)
BLUE = (0, 135, 255)
YELLOW = (255, 241, 192, 10)

class BasicSampleGenerator:
    def __init__(self, config):
        self.config = config
        self.params = {}
        config.assert_valid()

    def create_sample(self):
        size = self.config.background_image.size
        sample = self.create_background(size)
        score = random.randint(0, 100)

        if random.uniform(0, 1) > 0.5:
            text_colour = GREY
        else:
            text_colour = BLUE
        
        if score > 10:
            total_light_beams = random.randint(0, self.params.get("total_light_beams", 5))
            spread = math.pi/100
            y = size[1] - 10 

            for _ in range(total_light_beams):
                left = random.random() > 0.5
                x = -100 if left else size[0]+100
                angle = random.uniform(-math.pi/2, 0) if left else random.uniform(-math.pi, -math.pi/2)
                create_light_beam(sample, (x, y), angle, spread, YELLOW)

        create_ui(sample, self.config.background_image)
        create_score(sample, self.config.score_font, score, text_colour)        

        bounding_box = create_ball(sample, self.config.ball_image, (0, 360))

        for _ in range(random.randint(0, 4)):
            sample_type = random.uniform(0, 1)
            if sample_type < 0.3:
                rect = self.get_streaked_emotes(bounding_box[:2])
                populate_emotes(sample, self.config.emote_images, total=(0, 25), rect=rect)
            elif sample_type < 0.5:
                rect = self.get_local_scattered_emotes()
                populate_emotes(sample, self.config.emote_images, total=(0, 15), rect=rect)
            elif sample_type < 0.8:
                populate_emotes(sample, self.config.emote_images, total=(0, 10))
            else:
                pass
        
        label = bounding_box

        return (sample, label)

    def get_streaked_emotes(self, pos=None, x_offset=0.1, y_offset=0.1, width=0.05, height=0.3):
        if pos is None:
            x, y = random.uniform(0, 1), random.uniform(0, 1)
        else:
            x, y = pos

        left = x + random.uniform(-x_offset, x_offset)
        top = y + random.uniform(-y_offset, y_offset) - random.uniform(0, height)

        left = np.clip(left, 0, 1)
        top = np.clip(top, 0, 1)
        right = np.clip(left+random.uniform(0 ,width), 0, 1)
        bottom = np.clip(top+random.uniform(0, height), 0, 1)
        return (left, top, right, bottom)


    def get_local_scattered_emotes(self):
        left, top = random.random(), random.random()
        width, height = random.random(), random.random()
        right = np.clip(left+width, 0, 1)
        bottom = np.clip(top+height, 0, 1)
        return (left, top, right, bottom)

    def create_background(self, size, colour=(255, 255, 255, 255)):
        image = Image.new("RGBA", size, colour)
        return image






