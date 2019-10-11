import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont

ASSETS_PATH = "../../assets/"
IMG_PATH = os.path.join(ASSETS_PATH, "icons")
FONT_PATH = os.path.join(ASSETS_PATH, "fonts")
SAVE_PATH = os.path.join(ASSETS_PATH, "samples")
IMG_CACHE = {}
EMOJIS = []

for file in os.listdir(IMG_PATH):
    filepath = os.path.join(IMG_PATH, file)
    try:
        im = Image.open(filepath)
        IMG_CACHE[file] = im
        if 'success' in file or 'emote' in file:
            EMOJIS.append(im)
    except:
        pass

def create_background():
    blank_im = IMG_CACHE["blank.bmp"]
    background_im = blank_im.copy()
    return background_im

def create_score(canvas, score_range):
    font = ImageFont.truetype('segoeuil.ttf', 92)

    score = random.randint(*score_range)
    score_text = "{0}".format(score)
    score_width, score_height = font.getsize(score_text) 

    draw = ImageDraw.Draw(canvas)

    width, height = canvas.size

    x = int(width/2-score_width/2)
    y = int(height/5-score_height/2)

    draw.text((x, y), score_text, size=50, font=font, fill=(100, 100, 100))
    return canvas

def populate_emotes(canvas, total):
    lower, upper = total
    width, height = canvas.size

    total_emotes = random.randint(lower, upper)
    for _ in range(total_emotes):
        emote_img = random.choice(EMOJIS)
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        canvas.paste(emote_img, (x, y), emote_img)
    
    return canvas

def create_ball(canvas, rotation_range):
    ball_img = IMG_CACHE['ball.png']
    background_width, background_height = canvas.size
    ball_width, ball_height = ball_img.size

    x = random.randint(0, background_width-ball_width)
    y = random.randint(0, background_height-ball_height)
    rotation = random.randint(rotation_range[0], rotation_range[1])

    canvas.paste(ball_img.rotate(rotation), (x, y), ball_img)

    return canvas

def create_sample():
    img = create_background()
    create_score(img, (0, 100))
    create_ball(img, (0, 360))
    populate_emotes(img, (10, 100))
    return img

