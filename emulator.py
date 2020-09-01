import pygame as pg
import os
import math
import random

from src.emulator import *

ASSETS_DIR = "./assets/"

def main():

    SCREEN_WIDTH = 322
    SCREEN_HEIGHT = 455

    pg.init()
    surface = pg.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

    emulator = Emulator(ASSETS_DIR, Vec2D(SCREEN_WIDTH, SCREEN_HEIGHT))


    running = True
    clock = pg.time.Clock()

    while running:
        frame_ms = clock.tick(60)
        dt = frame_ms*1e-3

        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    emulator.on_click(*ev.pos)

        emulator.update(dt)
        surface.fill((255,255,255))
        emulator.render(surface)
        pg.display.flip()
    
    pg.quit()


if __name__ == '__main__':
    main()
    
