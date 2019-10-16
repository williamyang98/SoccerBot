import numpy as np
import random

from .overlays import *

class BasicSampleGenerator:
    def __init__(self, config):
        self.config = config
        config.assert_valid()
    
    def create_sample(self):
        sample = self.create_background()
        create_score(sample, self.config.score_font, (0, 100))        

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
                populate_emotes(sample, self.config.emote_images, total=(2, 10))
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

    def create_background(self):
        background_image = self.config.background_image.copy()
        return background_image






