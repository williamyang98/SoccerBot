import pygame as pg
import os
import math
import random

from src.emulator import *
from collections import deque

ASSETS_DIR = "./assets/"
ICONS_DIR = os.path.join(ASSETS_DIR, "icons")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

FONT_FILEPATH = os.path.join(FONTS_DIR, "segoeuil.ttf")

def clip(v, _min, _max):
    return max(min(v, _max), _min)

def main():

    SCREEN_WIDTH = 322
    SCREEN_HEIGHT = 455

    pg.init()

    surface = pg.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

    images = ImageLoader(ICONS_DIR)

    ball = Ball(images.ball)

    A_accel = Vec2D(0, 2000)

    clock = pg.time.Clock()

    high_score_counter = HighScoreCounter(FONT_FILEPATH, 18)
    high_score_counter.pos = Vec2D(SCREEN_WIDTH-12, 40)

    score_counter = ScoreCounter(FONT_FILEPATH)
    score_counter.pos = Vec2D(SCREEN_WIDTH//2, 85)

    emote_manager = EmoteManager(images)

    ball_spawn = Vec2D(SCREEN_WIDTH//2, SCREEN_HEIGHT-ball.radius-10)
    ball.pos = ball_spawn.copy()

    started = False
    score = 0
    high_score = 0

    running = True
    while running:
        frame_ms = clock.tick(60)
        dt = frame_ms*1e-3

        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    mouse_pos = Vec2D(*ev.pos)
                    max_random_x = 5
                    max_random_y = 5
                    emote_random_delta = Vec2D(
                        random.randint(-max_random_x, max_random_x), 
                        random.randint(-max_random_y, max_random_y))
                    emote_pos = mouse_pos + emote_random_delta

                    dist = (mouse_pos-ball.pos).length()
                    if dist < ball.radius:
                        started = True
                        score += 1

                        # high_score_counter.score = score 
                        score_counter.set_state(score, started)

                        min_bounce_vel = 900
                        max_bounce_vel = 2200
                        ball.vel.y = clip(ball.vel.y-min_bounce_vel, -max_bounce_vel, -min_bounce_vel)

                        horizontal_bounce = 450
                        horizontal_random = 150
                        horizontal_limit = 1000
                        
                        x_diff = -(mouse_pos.x-ball.pos.x)/ball.radius
                        dx_constant = x_diff * horizontal_bounce
                        dx_random = random.randint(-horizontal_random, horizontal_random)

                        ball.vel.x +=  dx_constant + dx_random
                        ball.vel.x = clip(ball.vel.x, -horizontal_limit, +horizontal_limit)

                        emote_manager.create_success(emote_pos)
                    else:
                        emote_manager.create_emote(emote_pos)
                

        # physics
        if started:
            C_drag = 0.01
            A_drag = -C_drag*ball.vel
            A_net = A_accel+A_drag

            ball.vel += A_net*dt
            ball.pos += ball.vel*dt

            if ball.pos.x-ball.radius < 0:
                ball.pos.x = ball.radius
                ball.vel.x = abs(ball.vel.x)
            elif ball.pos.x+ball.radius > SCREEN_WIDTH:
                ball.pos.x = SCREEN_WIDTH-ball.radius
                ball.vel.x = -abs(ball.vel.x)

            if ball.pos.y-ball.radius > SCREEN_HEIGHT:
                started = False
                ball.pos = ball_spawn.copy()
                high_score = max(score, high_score)
                high_score_counter.score = high_score
                score_counter.set_state(high_score, started)
                score = 0
                

        emote_manager.update(dt)

        # render
        surface.fill((255,255,255))
        high_score_counter.render(surface)
        score_counter.render(surface)

        ball.render(surface)
        emote_manager.render(surface)
        
        pg.display.flip()
    
    pg.quit()



class Emote:
    POPPING = 1
    STATIC = 2
    FADING = 3
    EXPIRED = 4

    def __init__(self, image, pos):
        # self.image = image.copy()
        self.image = image
        self.width, self.height = self.image.get_size()
        self.size = Vec2D(self.width, self.height)

        self.original_pos = pos.copy()
        self.pos = pos.copy()

        self.pop_duration = 0.2
        self.pop_distance = 40
        self.static_duration = 0.5
        self.fade_duration = 0.25 
        self.fade_distance = 150

        self.curr_duration = 0
        self.current_state = Emote.POPPING

        self.alpha = 0
    
    def update(self, dt):
        if self.current_state == Emote.EXPIRED:
            return

        self.curr_duration += dt
        if self.current_state == Emote.POPPING:
            prog = clip(self.curr_duration/self.pop_duration, 0, 1)
            self.alpha = int(prog*255)
            self.pos.y = self.original_pos.y - self.pop_distance * prog
            if self.curr_duration > self.pop_duration:
                self.curr_duration = self.pop_duration-self.curr_duration
                self.current_state = Emote.STATIC
        elif self.current_state == Emote.STATIC:
            self.alpha = 255
            self.pos.y = self.original_pos.y - self.pop_distance
            if self.curr_duration > self.static_duration:
                self.curr_duration = self.static_duration-self.curr_duration
                self.current_state = Emote.FADING
        elif self.current_state == Emote.FADING:
            prog = clip(self.curr_duration/self.fade_duration, 0, 1)
            self.alpha = int((1-prog)*255)
            self.pos.y = self.original_pos.y - self.pop_distance - self.fade_distance * prog
            if self.curr_duration > self.fade_duration:
                self.current_state = Emote.EXPIRED
    
    def render(self, surface):
        if self.current_state == Emote.EXPIRED:
            return
        # self.image.set_alpha(self.alpha)
        pos = (self.pos-self.size/2).cast_tuple(int)

        source = self.image.copy()

        image = pg.Surface(source.get_rect().size, pg.SRCALPHA)
        image.fill((255, 255, 255, self.alpha))
        source.blit(image, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        surface.blit(source, pos)

class EmoteManager:
    def __init__(self, images):
        self.images = images
        self.emotes = deque([])
    
    def create_success(self, pos):
        image = random.choice(self.images.success)
        self.emotes.append(Emote(image, pos))
    
    def create_emote(self, pos):
        image = random.choice(self.images.emotes)
        self.emotes.append(Emote(image, pos))
    
    def update(self, dt):
        for emote in self.emotes:
            emote.update(dt)
        
        while len(self.emotes) > 0 and self.emotes[0].current_state == Emote.EXPIRED:
            self.emotes.popleft()
    
    def render(self, surface):
        for emote in self.emotes:
            emote.render(surface)

class ScoreCounter:
    def __init__(self, font_path):
        self.small_font = pg.font.Font(font_path, 18)
        self.large_font = pg.font.Font(font_path, 75)

        self.primary_colour = (0,121,241)
        self.secondary_colour = (128,128,128)

        self.pos = Vec2D(0, 0)
        self.start_text = self.small_font.render("Current Best", True, self.secondary_colour)

        self.set_state(0, False)

        self.y_diff = 85

    def set_state(self, score, started):
        colour = self.primary_colour if not started else self.secondary_colour
        self.score_text = self.large_font.render(f"{score}", True, colour)
        self.started = started
    
    def render(self, surface):
        x, y = self.pos.x, self.pos.y
        if not self.started:
            width, height = self.start_text.get_size()
            rect = self.start_text.get_rect()
            rect.center = (x, y)
            surface.blit(self.start_text, rect)
            y_off = self.y_diff
        else:
            _, y_off = self.start_text.get_size()

        width, height = self.score_text.get_size()
        rect = self.score_text.get_rect()
        rect.center = (x, y+y_off)
        surface.blit(self.score_text, rect)

class HighScoreCounter:
    def __init__(self, font_path, size): 
        self.font = pg.font.Font(font_path, size)
        # self.font.set_bold(True)
        self.colour = (0,0,0)
        self.top_text = self.font.render("High Score", True, self.colour)

        self._score = 0 
        self.score_text = self.font.render(f"{self.score}", True, self.colour)

        self.y_diff = 25 

        self.pos = Vec2D(0, 0)

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score
        self.score_text = self.font.render(f"{self._score}", True, self.colour)
    
    def render(self, surface):
        x, y = self.pos.x, self.pos.y

        width, height = self.top_text.get_size()
        top_rect = self.top_text.get_rect()
        top_rect.center = (x-width//2, y)

        width, height = self.score_text.get_size()
        score_rect = self.score_text.get_rect()
        score_rect.center = (x-width//2, y+self.y_diff)

        surface.blit(self.top_text, top_rect)
        surface.blit(self.score_text, score_rect)


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

class ImageLoader:
    def __init__(self, directory):
        self.directory = directory

        self.ball = self.load("ball.png")
        self.emotes = [self.load(f"emote{i}.png") for i in range(1,6)]
        self.success = [self.load(f"success{i}.png") for i in range(1, 6)]
    
    def load(self, filename):
        return pg.image.load(os.path.join(self.directory, filename)).convert_alpha()

if __name__ == '__main__':
    main()
    

