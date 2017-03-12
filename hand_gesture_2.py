import cv2
import numpy as np

# Parameter Values
bgSubThreshold = 50
threshold = 60  # binary threshold
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
cv2.createTrackbar('trh1', 'Trackbar', threshold, 100, printThreshold)


while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'Trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)
    cv2.imshow("Input", frame)
    k = cv2.waitKey(10)
    if k == 27:     # esc button
        break
    if k == ord('b'):    # get the background
        fgbg = cv2.createBackgroundSubtractorKNN(0, bgSubThreshold)
        BgCaptured = 1
        print("Background captured. Proceed with hand segmentation")

