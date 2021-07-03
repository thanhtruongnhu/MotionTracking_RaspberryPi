import cv2
from tracker import *
from math import atan2, cos, sin, sqrt, pi
import numpy as np
import imutils


def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0])) # distance between p and q

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    ## [visualization1]


def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    rect = cv2.minAreaRect(c)
    # print('eigenvalues')
    # print(eigenvalues)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    # print('Center')
    # print(cntr)

    minAreaCntr = (int(rect[0][0]), int(rect[0][1]))
    # print('minAreaRect Center')
    # print(minAreaCntr)
    a = eigenvectors[0]
    b = np.array(np.subtract(np.array(minAreaCntr), np.array(cntr)))
    print(b)
    if (np.dot(a, b)<0):
        eigenvectors[0] = - eigenvectors[0]

    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

    p3 = (
        minAreaCntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        minAreaCntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)

    drawAxis(img, minAreaCntr, p3, (0, 0, 0), 7)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]

    # Label with the rotation angle
    # label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    label = "  Rotation Angle: " + str(int(np.rad2deg(angle))) + " degrees"
    cv2.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return angle


# Create tracker object
tracker = EuclideanDistTracker()

# cap = cv2.VideoCapture("highway.mp4")
cap = cv2.VideoCapture(0)

# Object detection from Stable camera
# object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()

    # Extract Region of interest
    # roi = frame

    # 1. Object Detection
    # mask = object_detector.apply(roi)


    # Downsize and maintain aspect ratio
    frame = imutils.resize(frame, width=400)
    cv2.imshow('Input Image', frame)

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    detections = []
    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv2.contourArea(c)

        # Ignore contours that are too small or too large
        if area < 1000 or 40000 < area:
            continue

        # Draw each contour only for visualisation purposes
        # cv2.drawContours(frame, contours, i, (0, 0, 255), 2)

        # Find the orientation of each shape
        getOrientation(c, frame)

        # # 2. Object Tracking
        # cv2.putText(frame, "object", (x, yQ - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


    # cv2.imshow("roi", roi)
    cv2.imshow('Output Image', frame)
    # cv2.waitKey(0)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()