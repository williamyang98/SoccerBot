import pygame as pg
import os
import math
import random
import numpy as np
import cv2
import os
import re
import glob
from argparse import ArgumentParser
from src.emulator import *

ASSETS_DIR = "./assets/"
DEFAULT_OUTPUT_DIR = "./assets/data/emulator_records/"

def main():
    parser = ArgumentParser()
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--record-size", type=int, default=10000)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    SCREEN_WIDTH = 322
    SCREEN_HEIGHT = 455

    TARGET_FPS = args.fps 

    pg.init()
    surface = pg.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    # surface = pg.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    emulator = Emulator(ASSETS_DIR, Vec2D(SCREEN_WIDTH, SCREEN_HEIGHT))
    emulator.firework_manager.max_fireworks = 12
    emulator.firework_manager.spawn_chance = 0.9 

    running = True
    total_frames = 0

    from create_tf_record_example import serialize_example

    def get_label():
        x, y = emulator.ball.pos.x, emulator.ball.pos.y
        x, y = x/SCREEN_WIDTH, y/SCREEN_HEIGHT

        if x < 0 or x > 1 or y < 0 or y > 1:
            x = 0
            y = 0
            confidence = 0
        else:
            confidence = 1
        return (x, y, confidence)
    
    show_preview = False

    with BatchedWriter(args.record_size, args.out_dir) as writer:
        while running:
            dt = 1/TARGET_FPS
            ball = emulator.ball

            x, y = emulator.ball.pos.x, emulator.ball.pos.y
            x, y = x/SCREEN_WIDTH, y/SCREEN_HEIGHT

            _, _, confidence = get_label() 

            # simulate ai
            if confidence:
                score = random.randint(0, 100000)
                emulator.high_score_counter.score = score
                emulator.score_counter.set_state(score, True)
                if ball.vel.y >= -10 and ball.pos.y > SCREEN_HEIGHT*0.7:
                    radius = int(ball.radius*0.8)
                    dx = random.randint(-radius, radius)
                    dy = random.randint(-radius, radius)
                    off = Vec2D(dx, dy)
                    off = off.norm() * radius
                    mouse_pos = ball.pos+off
                    emulator.on_click(mouse_pos.x, mouse_pos.y)
                elif ball.vel.y >= -100 and ball.pos.y > SCREEN_HEIGHT*0.6 and random.random() > 0.1:
                    radius = int(ball.radius * 1.1)
                    dx = random.randint(-radius, radius)
                    dy = random.randint(-radius, radius)
                    emulator.on_click(ball.pos.x+dx, ball.pos.y+dy)
                elif random.random() > 0.92:
                    radius = int(ball.radius * 1.5) 
                    dx = random.randint(-radius, radius)
                    dy = random.randint(-radius, radius)
                    emulator.on_click(ball.pos.x+dx, ball.pos.y+dy)
                elif random.random() > 0.8:
                    radius = int(ball.radius * 5) 
                    dx = random.randint(-radius, radius)
                    dy = random.randint(-radius, radius)
                    emulator.on_click(ball.pos.x+dx, ball.pos.y+dy)

            # extraneous button presses
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_p:
                        show_preview = not show_preview

            # tick
            emulator.update(dt)
            surface.fill((255,255,255))
            emulator.render(surface)

            image = pg.surfarray.array3d(surface)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, axes=[1,0,2])
            image = cv2.imencode(
                '.jpg', image, 
                (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
            
            x, y, confidence = get_label()
            example = serialize_example(
                image, 
                str.encode(f"f{total_frames}.jpg"),
                x,
                y,
                confidence)

            writer.write(example)
            total_frames += 1
            print(f"total frames: {total_frames}\r", end='')

            
            if show_preview:
                pg.display.flip()

    # closing
    print(f"\nwrote {total_frames} frames")

    if emulator.score != 0:
        emulator.on_fail()

    with open("gen_emulator.log", "a") as fp:
        fp.write("[session begin]\n")
        fp.write(f"deaths: {emulator.total_deaths}\n")
        fp.write(f"highscore: {emulator.highscore}\n")
        fp.write(f"scores: {','.join(map(str, emulator.scores))}\n")
        fp.write(f"clicks: {','.join(map(str, emulator.all_clicks))}\n")
        fp.write("\n")

class BatchedWriter:

    def __init__(self, max_records, out_dir):
        self.max_records = max_records
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)

        self.current_record = self.get_head()
        self.current_size = 0
        self.create_new()
    
    def get_head(self):
        X = glob.glob(os.path.join(self.out_dir, "images-*.tfrec"))
        p = re.compile(r".*images-(\d+)-\d+\.tfrec")
        res = map(p.findall, X)
        res = filter(lambda m: len(m) > 0, res)
        res = map(lambda m: int(m[0]), res)
        res = sorted(res)

        if len(res) > 0:
            head = res[-1] + 1
            print(f"moving head to record {head}")
            return head
        return 0

    def write(self, example):
        self.writer.write(example)
        self.current_size += 1

        if self.current_size >= self.max_records:
            self.close()
            self.current_size = 0
            self.current_record += 1
            self.create_new()
        
    def create_new(self):
        import tensorflow as tf
        filename = os.path.join(
            self.out_dir,
            f"images-{self.current_record}.tfrec")
        self.writer = tf.io.TFRecordWriter(filename)
    
    def close(self):
        self.writer.close()
        if self.current_size == 0:
            return

        old_name = os.path.join(
            self.out_dir,
            f"images-{self.current_record}.tfrec")
        new_name = os.path.join(
            self.out_dir, 
            f"images-{self.current_record}-{self.current_size}.tfrec")
        os.rename(old_name, new_name)
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    main()

