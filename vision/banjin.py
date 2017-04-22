#!/usr/bin/env python3

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw, ImageFont

# Convert numeric label to letter
def label2letter(label):
    if label == 0: # no tile
        return ' '
    elif label == 27: # blank tile
        return '_'
    elif 1 <= label <= 26: # letter tile
        return chr(label + 64)
    return '?' # unknown

# Convert letter to numeric label
def letter2label(letter):
    if letter == ' ':
        return 0
    elif letter == '_':
        return 27
    elif 'A' <= letter <= 'Z':
        return ord(letter) - 64
    return -1
    
# Return cutouts drawn on                                                                                                                                                                                   
def getCutouts(X, N, Y=None, border=2):
    s = int(np.ceil(np.sqrt(N)))

    # Create the output array
    out = np.zeros((X.shape[1], (X.shape[2] + border) * s - border,
                    (X.shape[3] + border) * s - border), dtype=np.uint8)
    color = (0, 0, 255)

    # For each square of cutouts
    for i in range(int(np.ceil(X.shape[0] / N)) + 1):

        # Aply each cutout to our cutout collection
        for j in range(min(N, X.shape[0] - i * N)):
            y = (j // s) * (X.shape[-2] + border)
            x = (j % s) * (X.shape[-1] + border)
            letter = str(i * N + j) if Y is None else label2letter(int(Y[i * N + j, 0]))
            out[:, y : y + X.shape[-2], x : x + X.shape[-1]] = X[i * N + j]
            cv2.putText(out[0], letter, (x, y + X.shape[-1] - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        if X.shape[1] == 1:
            yield out[0]
        else:
            yield out

