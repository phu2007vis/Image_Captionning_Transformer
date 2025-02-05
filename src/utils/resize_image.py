import math
from PIL import Image,ImageOps
import numpy as np
import torch

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    img = image

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.LANCZOS)
    
    img = ImageOps.pad(img, (image_height, image_height), color='white')
    return img

def resize_v2(w, h, size):
    new_w = int(size * float(w) / float(h))
    round_to = 10
    if h > w:
        new_w = math.ceil(new_w / round_to) * round_to
        size_h = size
    else:
        new_w = size
        size_h = int(size * float(h) / float(w))
        size_h = math.ceil(size_h / round_to) * round_to
    
    return new_w, size_h


class ProcessImageV2:
    def __init__(self, size,color = 'white'):
        self.size = size
        self.color = color 
    def __call__(self,image):
        
        size = self.size
        img = image

        w, h = img.size
        new_w, new_h = resize_v2(w, h, size)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        img = ImageOps.pad(img, (size, size), color= self.color)
        
        return img