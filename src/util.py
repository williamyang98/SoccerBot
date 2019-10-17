import cv2
import numpy as np

# image = numpy array 
def get_label_from_image(model, image, size):
    image = image[:,:,:3] / 255
    image = cv2.resize(image, size)
    # only get first 3 channels 
    label = model.predict(np.asarray([image]))[0] 
    return label

def map_bounding_box(bounding_box, shape):
    real_height, real_width = shape 
    x_centre_norm, y_centre_norm, width_norm, height_norm = bounding_box 

    x_centre = int(x_centre_norm*real_width)
    y_centre = int(y_centre_norm*real_height)
    width    = int(width_norm*real_width)
    height   = int(height_norm*real_height)

    return (x_centre, y_centre, width, height)

def draw_bounding_box(image, bounding_box, colour=(0, 20, 200)):
    mapped_bounding_box = map_bounding_box(bounding_box, image.shape[:2])
    x_centre, y_centre, width, height = mapped_bounding_box

    left = int(x_centre-width/2)
    right = int(x_centre+width/2)
    top = int(y_centre-height/2)
    bottom = int(y_centre+height/2)
    
    cv2.rectangle(image, (left, top), (right, bottom), colour, 2)
    return image