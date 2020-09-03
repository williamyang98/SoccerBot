import pygame as pg
import os
import math
import random
import numpy as np
import cv2
import os
import re
import glob

from src.emulator import *

ASSETS_DIR = "./assets/"
OUTPUT_DIR = "./assets/data/emulator_records/"

def main():
    SCREEN_WIDTH = 322
    SCREEN_HEIGHT = 455
    TARGET_FPS = 60

    pg.init()
    surface = pg.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    # surface = pg.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    emulator = Emulator(ASSETS_DIR, Vec2D(SCREEN_WIDTH, SCREEN_HEIGHT))
    emulator.firework_manager.max_fireworks = 12

    running = True

    total_frames = 0
    MAX_RECORD_SIZE = 10000
    record_size = 0
    current_record = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X = glob.glob(os.path.join(OUTPUT_DIR, "sample_*.tfrec"))
    p = re.compile(r".*sample_(\d+)_\d+\.tfrec")
    res = map(p.findall, X)
    res = filter(lambda m: len(m) > 0, res)
    res = map(lambda m: int(m[0]), res)
    res = sorted(res)

    if len(res) > 0:
        current_record = res[-1] + 1
        print(f"moving head to record {current_record}")

    record_filename = os.path.join(OUTPUT_DIR, f"sample_{current_record}.tfrec")

    import tensorflow as tf
    from create_tf_record_example import serialize_example

    writer = tf.io.TFRecordWriter(record_filename)

    def create_record():
        current_record += 1
        record_size = 0
        record_filename = os.path.join(OUTPUT_DIR, f"sample_{current_record}.tfrec")
        writer = tf.io.TFRecordWriter(record_filename)

    def save_record():
        writer.close()
        new_record_filename = os.path.join(OUTPUT_DIR, f"sample_{current_record}_{record_size}.tfrec")
        os.rename(record_filename, new_record_filename)

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

    try:
        while running:
            # frame_ms = clock.tick(TARGET_FPS)
            # dt = frame_ms*1e-3
            dt = 1/TARGET_FPS

            # ball_pos = emulator.ball.pos.copy()
            # click_pos = None

            ball = emulator.ball

            x, y = emulator.ball.pos.x, emulator.ball.pos.y
            x, y = x/SCREEN_WIDTH, y/SCREEN_HEIGHT

            _, _, confidence = get_label() 

            # simulate ai
            if confidence:
                if ball.vel.y >= -100 and ball.pos.y > SCREEN_HEIGHT*0.6 and random.random() > 0.15:
                    radius = int(ball.radius * 1.1)
                    dx = random.randint(-radius, radius)
                    dy = random.randint(-radius, radius)
                    emulator.on_click(ball.pos.x+dx, ball.pos.y+dy)
                elif random.random() > 0.95:
                    radius = int(ball.radius * 1.5) 
                    dx = random.randint(-radius, radius)
                    dy = random.randint(-radius, radius)
                    emulator.on_click(ball.pos.x+dx, ball.pos.y+dy)
                elif random.random() > 0.9:
                    radius = int(ball.radius * 5) 
                    dx = random.randint(-radius, radius)
                    dy = random.randint(-radius, radius)
                    emulator.on_click(ball.pos.x+dx, ball.pos.y+dy)

            # extraneous button presses
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False

            # tick
            emulator.update(dt)
            surface.fill((255,255,255))
            emulator.render(surface)

            image = pg.surfarray.array3d(surface)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, axes=[1,0,2])
            
            image = cv2.imencode(
                '.jpg', image, 
                (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
            
            x, y, confidence = get_label()
            example = serialize_example(
                image, 
                str.encode(f"FRAME_{total_frames}.jpg"),
                x,
                y,
                confidence)

            writer.write(example)

            total_frames += 1
            record_size += 1

            print(f"total frames: {total_frames}\r", end='')

            if record_size >= MAX_RECORD_SIZE:
                save_record()
                create_record()
    except KeyboardInterrupt:
        pass
    finally:
        # closing
        print(f"\nwrote {total_frames} frames")

        if record_size > 0:
            save_record()

if __name__ == '__main__':
    main()

