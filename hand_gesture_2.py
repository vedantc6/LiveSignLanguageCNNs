import cv2
import numpy as np

# Parameter Values
bgSubThreshold = 50
threshold = 170  # binary threshold
cap_region_x_begin = 0.55
cap_region_y_end = 0.8
blurValue = 41  # gaussian blur

# Variables
BgCaptured = 0


def printThreshold(thr):
    print("Threshold value should be changed to " + str(thr))


def removeBG(frame):
    fgmask = fgbg.apply(frame)
    kernel = np.ones((3,3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# Camera Initialization
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('Trackbar')
cv2.createTrackbar('trh1', 'Trackbar', threshold, 255, printThreshold)


while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'Trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (int(cap_region_x_begin*frame.shape[1]), 0),
                    (frame.shape[1], int(cap_region_y_end*frame.shape[0])),
                    (255, 0, 0), 2)
    cv2.imshow("Input", frame)
    if BgCaptured == 1:
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end*frame.shape[0]),
                int(cap_region_x_begin*frame.shape[1]):frame.shape[1]]  # clip ROI
        cv2.imshow('Mask', img)
        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('Blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('Threshold', thresh)
    k = cv2.waitKey(10)
    if k == 27:     # esc button
        break
    if k == ord('b'):    # get the background
        fgbg = cv2.createBackgroundSubtractorKNN(0, bgSubThreshold)
        BgCaptured = 1
        print("Background captured. Proceed with hand segmentation")

