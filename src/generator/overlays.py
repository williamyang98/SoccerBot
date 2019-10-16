import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont

def create_score(canvas, font, score_range, fill):
    score = random.randint(*score_range)
    score_text = "{0}".format(score)
    score_width, score_height = font.getsize(score_text) 
    width, height = canvas.size

    x = int(width/2-score_width/2)
    y = int(height/5-score_height/2)

    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), score_text, size=50, font=font, fill=fill)

def populate_emotes(canvas, emotes, total, rect=(0, 0, 1, 1)):
    lower, upper = total
    width, height = canvas.size

    left, top, right, bottom = rect
    left, right = left*width, right*width
    top, bottom = top*height, bottom*height

    total_emotes = random.randint(lower, upper)
    for _ in range(total_emotes):
        emote_img = random.choice(emotes)
        x = int(random.uniform(left, right))
        y = int(random.uniform(top, bottom))
        canvas.paste(emote_img, (x, y), emote_img)
    
# x_centre, y_centre, width, height - normalised to dimensions
def create_ball(canvas, ball, rotation_range):
    background_width, background_height = canvas.size
    ball_width, ball_height = ball.size

    x = random.randint(0, background_width-ball_width)
    y = random.randint(0, background_height-ball_height)
    rotation = random.randint(rotation_range[0], rotation_range[1])

    canvas.paste(ball.rotate(rotation), (x, y), ball)

    x_centre, y_centre = x+ball_width/2, y+ball_height/2    
    x_centre_norm, y_centre_norm = x_centre/background_width, y_centre/background_height
    width_norm, height_norm = ball_width/background_width, ball_height/background_height

    return (x_centre_norm, y_centre_norm, width_norm, height_norm)