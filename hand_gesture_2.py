import cv2
import numpy as np
import copy
import math
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
# print(ROOT)

# Parameter Values
bgSubThreshold = 60
threshold = 70  # binary threshold
cap_region_x_begin = 0.55
cap_region_y_end = 0.8
blurValue = 41  # gaussian blur

# Variables
BgCaptured = 0
c = 0

def printThreshold(thr):
    print("Threshold value should be changed to " + str(thr))


def removeBG(frame):
    fgmask = fgbg.apply(frame)
    kernel = np.ones((3,3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1) #strips out the outermost layer of pixels in a structure
    res = cv2.bitwise_and(frame, frame, mask=fgmask)  #a bitwise of the functin wth itself
    return res


def calculateFingers(res, drawing):
    # convexity defect
    hull = cv2.convexHull(res, returnPoints=False) #to find convexity defects , returns convex defects not contour values
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):     # avoid crash
            cnt = 0
            for i in range(defects.shape[0]):   # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
        return False, 0

# Camera Initialization
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('Trackbar')
cv2.createTrackbar('trh1', 'Trackbar', threshold, 255, printThreshold)


while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'Trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter, keeps edges intact but smooths it over
    frame = cv2.flip(frame, 1) # flips over y-axis
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
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY) #simple threshold function like signum, can be applied only on grayscale images
        cv2.imshow('Threshold', thresh)

        # getting contours
        thresh1 = copy.deepcopy(thresh)
        _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):     # find the biggest contour (area wise)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
            res = contours[ci]  # res contains the biggest contour
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal, cnt = calculateFingers(res, drawing)
        cv2.imshow('Output', drawing)

    k = cv2.waitKey(10)
    if k == 27:     # esc button
        break
    if k == ord('b'):    # get the background
        fgbg = cv2.createBackgroundSubtractorKNN(0, bgSubThreshold+20)
        BgCaptured = 1
        print("Background captured. Proceed with hand segmentation")
    elif k == ord('r'):     # reset the background
        fgbg = None
        BgCaptured = 0
        print("Background is reset. Press b to capture the background again.")
    elif k == ord('c'):     # click and save the picture
        c += 1
        UPLOAD_DIR = os.path.join(ROOT + '/' + 'uploads')
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
        cv2.imwrite(UPLOAD_DIR + '/' + str(c) + '.png', blur)
        print("Image captured and sent for testing.")






