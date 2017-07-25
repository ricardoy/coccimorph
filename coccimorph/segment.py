import numpy as np
import cv2
import argparse

vx = []
vy = []


def binaryze(img, threshold):
    img_grayscale = np.average(img, axis=2)
    binarizer = np.vectorize(lambda x: 255 if x < threshold else 0)
    img_bin = binarizer(img_grayscale)
    return img_bin


def segment(filename, threshold, scale=None):
    img = cv2.imread(filename)
    height, width, _ = img.shape
    if scale is not None:
        width = int(width/(scale/11.))
        height = int(height/(scale/11.))
        img = cv2.resize(img, (width, height))

    img_bin = binaryze(img, threshold)
    cv2.imwrite('/tmp/a.png', img_bin)
    print(img_bin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment image.')
    parser.add_argument('-input-file', type=str)
    parser.add_argument('-threshold', type=int)
    parser.add_argument('-scale', type=float)
    args = parser.parse_args()
    if args.input_file is not None and args.threshold is not None:
        segment(args.input_file, args.threshold, args.scale)
