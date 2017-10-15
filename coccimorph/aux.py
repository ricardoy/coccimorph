import cv2
import numpy as np


def load_image(filename, scale):
    img = cv2.imread(filename)
    height, width, _ = img.shape
    if scale is not None:
        width = int(width / (scale / 11.))
        height = int(height / (scale / 11.))
        img = cv2.resize(img, (width, height))
    return np.transpose(img, axes=[1, 0, 2])