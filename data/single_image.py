import torch
import cv2
import numpy as np

def open_image(img_path):

    image = cv2.imread(img_path)
    
    # Reshape to pytorch channel order
    w, h = image.shape[0], image.shape[1]
    image = np.reshape(image, (3, w, h))
    
    return torch.from_numpy(image).float()
