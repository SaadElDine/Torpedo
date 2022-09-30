import cv2
import numpy as np

#Choose From The Test Images Included
image = cv2.imread('Test 0.png')

def detectColors():    #TRACKBARS TO DETECT LOWER AND UPPER YELLOW AND RED Using Mask
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

        cv2.imshow("Original Yellow", image)
        cv2.imshow("HSV Yellow", imgHSV)
        cv2.imshow("Result Car", result)
        if cv2.waitKey(1) == 30:
            break
    cv2.destroyAllWindows()

def isYellowSquare(image):  #Detect Yellow Square
    lowerYellow = np.array([16, 87, 0]).reshape((1, 1, 3))    #from Trackbars
    upperYellow = np.array([41, 255, 255]).reshape((1, 1, 3))
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imageHSV, lowerYellow, upperYellow)
    count = np.count_nonzero(mask)
    if count > 6000:
        return True
    return False

def isRedStar(image):      #Detect Red Star
    lowerRed = np.array([141, 70, 0]).reshape((1, 1, 3))
    upperRed = np.array([200, 255, 241]).reshape((1, 1, 3))
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imageHSV, lowerRed, upperRed)
    count = np.count_nonzero(mask)
    if count > 6000:
        return True
    return False

firstPoint = None
secondPoint = None
croppedImage = None

def crop(event, x, y, flag, param):     #crop the image
    global firstPoint, secondPoint, croppedImage
    if event == cv2.EVENT_LBUTTONDOWN:
        if firstPoint is None:
            firstPoint = (x, y)
        elif secondPoint is None:
            secondPoint = (x, y)
            minY = min(firstPoint[1], secondPoint[1])
            maxY = max(firstPoint[1], secondPoint[1])
            if minY != maxY:
                croppedImage = image[minY:maxY, :]
            firstPoint = None
            secondPoint = None

cv2.namedWindow('Image')
cv2.setMouseCallback("Image", crop)

while True:
    cv2.imshow('Image', image)
    if croppedImage is not None:
        croppedHSV = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
        cv2.imshow('You Choosen Image After Crop ', croppedImage)
        width = int(croppedImage.shape[1] / 3)
        firstSquare = croppedImage[:, :width]
        secondSquare = croppedImage[:, width: width * 2]
        thirdSquare = croppedImage[:, width * 2:]
        yellowText = "Yellow Square Detected!!"
        redText = "Red Star Detected!!"
        if isYellowSquare(firstSquare):
            firstSquare = cv2.putText(firstSquare, yellowText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        if isYellowSquare(secondSquare):
            secondSquare = cv2.putText(secondSquare, yellowText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        if isYellowSquare(thirdSquare):
            thirdSquare = cv2.putText(thirdSquare, yellowText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        if isRedStar(firstSquare):
            firstSquare = cv2.putText(firstSquare, redText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        if isRedStar(secondSquare):
            secondSquare = cv2.putText(secondSquare, redText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        if isRedStar(thirdSquare):
            thirdSquare = cv2.putText(thirdSquare, redText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("First Square", firstSquare)
        cv2.imshow("Second Square", secondSquare)
        cv2.imshow("Third Square", thirdSquare)
        croppedImage = None
    if cv2.waitKey(1) & 0xFF == 30:
        break
cv2.destroyAllWindows()
