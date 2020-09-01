import math
from .Vec2D import Vec2D

def clip(v, _min, _max):
    return max(min(v, _max), _min)

def point_rot(p, sin, cos=None):
    if cos is None:
        sin, cos = math.sin(sin), math.cos(sin)

    x = cos*p.x - sin*p.y
    y = sin*p.x + cos*p.y
    return Vec2D(x, y)