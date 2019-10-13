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
        populate_emotes(sample, self.config.emote_images, (10, 100))
        return (sample, bounding_box)

    def create_background(self):
        background_image = self.config.background_image.copy()
        return background_image






