from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import time
import serial
import math

# global variables

# rf send

ser = serial.Serial(
    port='/dev/ttyAMA0',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)


# function convert RGB value to HSV value

def convertHSV(G, R, B):
    colorBGR = np.uint8([[[B, G, R]]])
    colorHSV = cv2.cvtColor(colorBGR, cv2.COLOR_BGR2HSV)
    H_value = colorHSV[:, 0, 0]
    a = int(H_value)
    return a


# function standardixed coordinate axis 2D image coordinate to 3D world coordinate with Z=1

def findCoordinate_rb1():
    coordinate = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    res1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if x > 10:
        up_yellow1 = np.array([x - 10, 100, 100])
        under_yellow1 = np.array([x + 10, 255, 255])
    else:

        up_yellow1 = np.array([0, 100, 100])
        under_yellow1 = np.array([x + 10, 255, 255])
    mask1 = cv2.inRange(hsv, up_yellow1, under_yellow1)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(ret) > 0:
        c = ret[0]
        ((x, y), R) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        R = int(R)
        coordinate.append(x)
        coordinate.append(y)
        coordinate.append(R)
    else:
        coordinate = None
    return coordinate


# calibration
myfile = np.load('calib.npz')
mtx = myfile['mtx']
dist = myfile['dist']
newmtx = myfile['newcameramtx']

# Stream video from module Pi camera
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 40
rawCapture = PiRGBArray(camera, size=(320, 240))

time.sleep(1)

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    # start = time.time()
    img = frame.array
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # time.sleep(1)
    x, y, h, w = roi
    img = img[y:y + w, x:x + h]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # rb_1

    up_yellow1 = np.array([0, 100, 100])
    under_yellow1 = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, up_yellow1, under_yellow1)

    kernel = np.ones((5, 5), np.uint8)
    mask_rb1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    mask_rb1 = cv2.morphologyEx(mask_rb1, cv2.MORPH_OPEN, kernel)

    ret1 = cv2.findContours(mask_rb1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if (len(ret1) > 0):
        c_1 = ret1[0]
        # rb1
        ((x_1, y_1), R_1) = cv2.minEnclosingCircle(c_1)
        M = cv2.moments(c_1)
        x_1 = int(M['m10'] / M['m00'])
        y_1 = int(M['m01'] / M['m00'])
        center1 = (x_1, y_1)
        R_1 = int(R_1)

        # process on map
        if ((x_1 >= 33 and y_1 >= 16)):
            x_world1 = (x_1 - 33) * (200 / 28)
            y_world1 = (y_1 - 16) * (200 / 28)
            # print (int(x_world1),int(y_world1))
            '''
                if R>10:
                    cv2.circle(img,center,R,(255,0,0),2)
                    cv2.putText(img,"("+str(x_world)+","+str(y_world)+")",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
    '''
            # process coordinate rb1
            canny_rb1 = c_1[:, :, 0]
            # print a
            max_a1 = []
            for i in range(0, len(canny_rb1)):
                k_1 = math.pow((int(c_1[i, :, 0]) - x_1), 2) + math.pow((int(c_1[i, :, 1]) - y_1), 2)
                max_a1.append(k_1)
                j_1 = max_a1.index(max(max_a1))
            x_max1 = int(c_1[j_1, :, 0])
            y_max1 = int(c_1[j_1, :, 1])
            # print (x_max1,y_max1)
            u_1 = np.array([x_max1 - x_1, y_max1 - y_1])
            v_1 = np.array([1, 0])
            u_value1 = math.sqrt(math.pow(u_1[0], 2) + math.pow(u_1[1], 2))
            v_value1 = 1
            rad = math.acos(np.sum(u_1 * v_1) / (u_value1 * v_value1))
            if (y_max1 >= y_1):
                Angle1 = math.degrees(rad)
            if (y_max1 < y_1):
                Angle1 = 360 - math.degrees(rad)
            print(Angle1)

            if R_1 > 1:
                cv2.circle(img, center1, 2, (255, 255, 255), 2)
                cv2.circle(img, (x_max1, y_max1), 2, (255, 255, 255), 2)
                # cv2.putText(img,"("+str(x_1)+","+str(y_1)+")",(x_1,y_1),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

        else:
            cv2.imshow('frame', img)

        # RF send

    byte_0 = int(255)  # Start Byte
    # current position
    byte_1 = int(x_world1) // 254  # Transfer the Quotient
    byte_2 = int(x_world1) % 254  # Transfer the Remainder
    byte_3 = int(y_world1) // 254  # Transfer the Quotient
    byte_4 = int(y_world1) % 254  # Transfer the Remainder
    # target positon
    # current angle
    byte_5 = int(Angle1) // 254
    byte_6 = int(Angle1) % 254
    byte_7 = int(255)  # Stop Byte
    # list1=[byte_0]
    list1 = [byte_0, byte_1, byte_2, byte_3, byte_4, byte_5, byte_6, byte_7]
    # cv2.imshow('frame',img)

    ser.write(list1)
    # stop = time.time()
    # print (stop - start)
    ser.flush()
    # print(list1)
    # time.sleep(0.25)

    cv2.imshow('frame', img)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)
    if key == ord("q"):
        ser.close()
        cv2.destroyAllWindows()
        break



