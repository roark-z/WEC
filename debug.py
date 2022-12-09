'''
Debug file for testing code, not intended to become the training/testing file
'''

import os 
import sys

# Add models
importpath = os.path.join(os.getcwd(), 'model')
sys.path.append(importpath)

# Add data
importpath = os.path.join(os.getcwd(), 'data')
sys.path.append(importpath)


from model import FrameEncoder
from single_image import open_image

# Make a network
FEncoder = FrameEncoder((420, 640), 64)

# Load an image
img_path = 'Trashmania_2.png'
image = open_image(img_path)

# Process the image
out_vector = FEncoder(image)


