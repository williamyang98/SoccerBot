import pygame as pg
import math
import numpy as np
from .Vec2D import Vec2D
from .util import point_rot, clip, get_points

class Firework:
    FLYING = 1
    EXPLODING = 2
    FINISHED = 3

    def __init__(self, start, end, ToF, explosion_colour):
        self.start = start
        self.end = end
        self.flight_distance = (start-end).length()
        self.pos = start.copy()

        self.ToF = ToF
        self.elapsed_ToF = 0

        self.vel = (end-start)/ToF
        self.state = Firework.FLYING

        self.explosion_duration = 0.6
        self.elapsed_explosion_duration = 0

        self.streak_colour = (155, 155, 155)
        self.explosion_colour = explosion_colour
        # self.explosion_colour = (0, 255, 0)
    
    def is_finished(self):
        return (self.state == Firework.FINISHED)
    
    def update(self, dt):
        if self.is_finished():
            return
        
        if self.state == Firework.FLYING:
            self.pos += self.vel*dt
            self.elapsed_ToF = clip(self.elapsed_ToF+dt, 0, self.ToF)
            if self.elapsed_ToF >= self.ToF:
                self.state = Firework.EXPLODING
            return
        
        if self.state == Firework.EXPLODING:
            self.elapsed_explosion_duration = clip(
                self.elapsed_explosion_duration+dt, 
                0, self.explosion_duration)
            if self.elapsed_explosion_duration >= self.explosion_duration:
                self.state = Firework.FINISHED

    def render(self, surface):
        if self.is_finished():
            return

        if self.state == Firework.FLYING:
            self.render_streak(surface)
        elif self.state == Firework.EXPLODING:
            self.render_explosion(surface)
    
    def render_explosion(self, surface):
        self.render_fireball(surface)
        self.render_rays(surface)

    def render_rays(self, surface):
        prog = self.elapsed_explosion_duration / self.explosion_duration
        # center of explosion
        pos = self.end

        alpha = int((1-prog)*255)

        max_length = 100

        offset = max_length * 0.2
        upper = max_length * prog + offset
        lower = max_length * 0.3 * prog + offset

        total = 8
        angles = np.linspace(0, 1, total+1)[:-1] * math.pi * 2

        ray_dirs = [point_rot(Vec2D(0, 1), alpha) for alpha in angles]

        image = pg.Surface(surface.get_size())
        image.set_colorkey((0, 0, 0))
        image.set_alpha(alpha)

        for alpha in angles:
            dim = Vec2D(8, upper-lower)
            ray_dir = point_rot(Vec2D(0, 1), alpha)
            ray_pos = pos + ray_dir*(upper-lower)/2

            points = get_points(ray_pos, alpha, dim)
            points = [p.cast_tuple(int) for p in points]

            pg.draw.polygon(image, self.explosion_colour, points)
            # pg.draw.line(
            #     image, self.explosion_colour, 
            #     p0.cast_tuple(int), p1.cast_tuple(int), 8)
        
        surface.blit(image, (0, 0))
    
    def render_fireball(self, surface):
        prog = self.elapsed_explosion_duration / self.explosion_duration
        # center of explosion
        pos = self.end

        alpha = int((1-prog)*255)

        max_explosion_radius = 50

        K_min = 0.2
        K = (1-prog)*(1-K_min) + K_min
        explosion_radius = max_explosion_radius * K

        image = pg.Surface(surface.get_size())
        image.set_colorkey((0, 0, 0))
        image.set_alpha(alpha)

        pg.draw.circle(
            image, self.explosion_colour,
            pos.cast_tuple(int), int(explosion_radius))

        surface.blit(image, (0, 0))

    
    def render_streak(self, surface):
        prog = self.elapsed_ToF / self.ToF
        K = math.cos(prog * math.pi * 2)
        K = K/2 + 0.5
        K = 1-K
        # K = min(0.8, K)
        # K = 1-abs(prog-0.5)*2

        direction = self.end-self.start
        p0 = self.start + direction*prog

        length = K*self.flight_distance*0.25

        p1 = p0 + direction.norm()*-length

        pg.draw.line(
            surface, self.streak_colour, 
            p0.cast_tuple(int), p1.cast_tuple(int), 3)