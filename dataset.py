import cv2
import os
import train
import grayscale
import new_model

ROOT = os.path.dirname(os.path.abspath(__file__))
# print(ROOT)

# Parameter Values
bgSubThreshold = 60
threshold = 70  # binary threshold
cap_region_x_begin = 0.665
cap_region_y_end = 0.6

# Variables
BgCaptured = 0
c = 0

classes = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

value = 0

def printThreshold(thr):
    print("Threshold value should be changed to " + str(thr))


# def imagemul(img, i, value):
#     val = classes[value]
#     if not os.path.exists(UPLOAD_DIR + '/' + val + '/'):
#         os.makedirs(UPLOAD_DIR + '/' + val + '/')
#     cv2.imwrite(UPLOAD_DIR + '/' + val + '/' + str(i) + '.png', img)
#     i += 1
#     frm = cv2.flip(img, 1)
#     cv2.imwrite(UPLOAD_DIR + '/' + val + '/' + str(i) + '.png', frm)
#     i = i + 1
#     return i

# Camera Initialization
camera = cv2.VideoCapture(0)
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
        # img = removeBG(frame)
        frame = frame[0:int(cap_region_y_end*frame.shape[0]),
                int(cap_region_x_begin*frame.shape[1]):frame.shape[1]]  # clip ROI

    k = cv2.waitKey(10)
    if k == 27:     # esc button
        break
    if k == ord('b'):    # get the background
        fgbg = cv2.createBackgroundSubtractorKNN(0, bgSubThreshold+20)
        BgCaptured = 1
        print("Background captured. Proceed with hand segmentation")
    elif k == ord('r'):     # reset the background
        value += 1
        c = 0
        fgbg = None
        BgCaptured = 0
        print("Background is reset. Press b to capture the background again.")
    elif k == ord('c'):     # click and save the picture
        c += 1
        print(c)
        # UPLOAD_DIR = os.path.join(ROOT)
        # if not os.path.exists(UPLOAD_DIR):
        #     os.makedirs(UPLOAD_DIR)
        # c = imagemul(frame, c, value)

        print("Image captured and sent for testing.")
        cv2.imwrite('test_image.png', frame)
        grayscale.grey()
        new_model.test_model()
        # train.test_model()
        # value = train.test_model()
        # print(value)
        # cv2.putText(frame, str(value), (50,50), cv2.FONT_ITALIC, 4, (255,255,255), 2, cv2.LINE_AA)






