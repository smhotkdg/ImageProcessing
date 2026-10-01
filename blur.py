import cv2

import numpy as np

color = cv2.imread('noisy_image.png')

avg = cv2.GaussianBlur(color, (5, 5), 0)

cv2.imshow('color', color)
cv2.imshow('out', avg)
cv2.waitKey(0)