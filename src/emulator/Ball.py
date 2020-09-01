from .Vec2D import Vec2D

class Ball:
    def __init__(self, image):
        self.image = image
        self.width, self.height = self.image.get_size()
        self.radius = (self.width+self.height)/4

        self.size = Vec2D(self.width, self.height)

        self.pos = Vec2D(0, 0)
        self.vel = Vec2D(0, 0)

    def render(self, surface):
        surface.blit(self.image, (self.pos-self.size/2).cast_tuple(int))