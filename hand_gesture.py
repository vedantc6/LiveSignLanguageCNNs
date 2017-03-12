import cv2
import numpy as np

# creating camera object
cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
fgbg = cv2.createBackgroundSubtractorKNN()

while cap.isOpened():
    # reading the frames
    ret, img = cap.read()
    fgmask = fgbg.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # using gaussian blur function
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # ret, thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # # displaying the frames
    cv2.imshow('input', fgmask)
    k = cv2.waitKey(10)
    if k == 27:     # esc button corresponds to 27 value.
        break




