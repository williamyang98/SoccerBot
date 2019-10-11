import tensorflow as tf
from tensorflow.keras import backend as K

# IOU = (A ∩ B) / (A u B)
# The ratio of the intersection against the union
# if the boxes are far apart, then high union, and low intersection (IOU decreases)
# if the boxes are near, then lower union, and higher intersection (IOU increase)
def calculate_IOU(y_target, y_expected):
    # get intersection rectangle
    a_centre_x, a_centre_y, a_width, a_height = y_target[...,0], y_target[...,1], y_target[...,2], y_target[...,3] 
    b_centre_x, b_centre_y, b_width, b_height = y_expected[...,0], y_expected[...,1], y_expected[...,2], y_expected[...,3] 

    left   = K.maximum(a_centre_x-a_width/2.0, b_centre_x-b_width/2.0) 
    right  = K.minimum(a_centre_x+a_width/2.0, b_centre_x+b_width/2.0)
    top    = K.maximum(a_centre_y-a_height/2.0, b_centre_y-b_height/2.0)
    bottom = K.minimum(a_centre_y+a_height/2.0, b_centre_y+b_height/2.0)

    intersect_area = K.maximum(0.0, right-left)*K.maximum(0.0, bottom-top)
    union_area = (a_width*a_height) + (b_width*b_height) - intersect_area
    IOU = intersect_area/union_area
    return IOU

def calculate_loss(y_target, y_predicted):
    mean_square_error = tf.losses.mean_squared_error(y_target, y_predicted)
    IOU = calculate_IOU(y_target, y_predicted)
    loss = mean_square_error + (1-IOU)
    return loss
    

