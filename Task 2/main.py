import numpy as np
import cv2
image = cv2.imread('508.jfif')

def firstTask(image):
    grayCar = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('uint16')
    grayCar0 = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3
    image = image.astype('uint8')
    grayCar0 = grayCar0.astype('uint8')
    cv2.imshow("Original Car", image)
    cv2.imshow("Gray Car Using Built In Fn ", grayCar)
    cv2.imshow("Gray Car Using My Fn ", grayCar0)
    cv2.waitKey(0)

def secondTask():
    def fn(x):
        pass
    def builtTrackBars():
        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars", 640, 240)
        cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, fn)
        cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, fn)
        cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, fn)
        cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, fn)
        cv2.createTrackbar("Val Min", "TrackBars", 0, 255, fn)
        cv2.createTrackbar("Val Max", "TrackBars", 255, 255, fn)

    builtTrackBars()
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    while True:
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        print(h_min, h_max, s_min, s_max, v_min, v_max)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        result = cv2.bitwise_and(image, image, mask = mask)

        cv2.imshow("Original Car", image)
        cv2.imshow("HSV Car", imgHSV)
        cv2.imshow("Mask Car", mask)
        cv2.imshow("Result Car", result)
        if cv2.waitKey(1) == 30:
            break
    cv2.destroyAllWindows()


while True:
    number = int(input("Please Enter Number 1 to run task 1 (Convert to Gray), Number 2 to run task 2:"))
    if number == 1:
        firstTask(image)
    elif number == 2:
        secondTask()
    else:
        break

