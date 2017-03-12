import cv2
import numpy as np

# creating camera object
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # reading the frames
    ret, img = cap.read()
    # displaying the frames
    cv2.imshow('input', img)
    k = cv2.waitKey(10)
    if k == 27:     # esc button corresponds to 27 value.
        break




