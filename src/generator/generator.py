import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont
from paths import IMG_PATH, ASSETS_PATH

IMG_CACHE = {}
EMOJIS = []

font = ImageFont.truetype(os.path.join(ASSETS_PATH, "fonts", "segoeuil.ttf"), 92)

for file in os.listdir(IMG_PATH):
    filepath = os.path.join(IMG_PATH, file)
    try:
        im = Image.open(filepath)
        IMG_CACHE[file] = im
        if 'success' in file or 'emote' in file:
            EMOJIS.append(im)
    except:
        pass

blank_img = IMG_CACHE["blank.bmp"]
ball_img = IMG_CACHE['ball.png']

def create_sample():
    img = create_background()
    create_score(img, (0, 100))
    bounding_box = create_ball(img, (0, 360))
    populate_emotes(img, (10, 100))
    return (img, bounding_box)

def create_background():
    background_im = blank_img.copy()
    return background_im

def create_score(canvas, score_range):
    score = random.randint(*score_range)
    score_text = "{0}".format(score)
    score_width, score_height = font.getsize(score_text) 
    width, height = canvas.size

    x = int(width/2-score_width/2)
    y = int(height/5-score_height/2)

    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), score_text, size=50, font=font, fill=(100, 100, 100))

def populate_emotes(canvas, total):
    lower, upper = total
    width, height = canvas.size

    total_emotes = random.randint(lower, upper)
    for _ in range(total_emotes):
        emote_img = random.choice(EMOJIS)
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        canvas.paste(emote_img, (x, y), emote_img)
    
# x_centre, y_centre, width, height - normalised to dimensions
def create_ball(canvas, rotation_range):
    background_width, background_height = canvas.size
    ball_width, ball_height = ball_img.size

    x = random.randint(0, background_width-ball_width)
    y = random.randint(0, background_height-ball_height)
    rotation = random.randint(rotation_range[0], rotation_range[1])

    canvas.paste(ball_img.rotate(rotation), (x, y), ball_img)

    x_centre, y_centre = x+ball_width/2, y+ball_height/2    
    x_centre_norm, y_centre_norm = x_centre/background_width, y_centre/background_height
    width_norm, height_norm = ball_width/background_width, ball_height/background_height

    return (x_centre_norm, y_centre_norm, width_norm, height_norm)




