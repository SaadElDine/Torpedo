import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

image1 = cv.imread('panda 1.jfif', 0)
image1 = cv.resize(image1, (350, 350))
image2 = cv.imread('panda 2.jfif', 0)
image2 = cv.resize(image2, (350, 350))
image3 = cv.imread('panda 3.jfif', 0)
image3 = cv.resize(image3, (350, 350))
image4 = cv.imread('panda 4.jfif', 0)
image4 = cv.resize(image4, (350, 350))
image5 = cv.imread('boat.jfif', 0)
image5 = cv.resize(image5, (350, 350))
image6 = cv.imread('Task.jfif', 0)
image6 = cv.resize(image6, (350, 350))


def displayImage(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def display2Images(img1, img2):
    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title("Before")

    fig.add_subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title("After")
    plt.show()


def firstTask(img1, img2, img3, img4):
    imageA = np.concatenate((img1, img2), axis=1)
    imageB = np.concatenate((img3, img4), axis=1)
    image = np.concatenate((imageA, imageB), axis=0)
    displayImage(image)


firstTask(image1, image2, image3, image4)


def secondTask(img):  # To Zero Threshold With numpy
    img1 = img.copy()
    pixel = int(input("Enter Your Required Pixel Value (0->255): "))
    img1[img1 < pixel] = 0  # Boolean Indexing
    display2Images(img, img1)

secondTask(image5)
ret, thresh1 = cv.threshold(image5, 120, 255, cv.THRESH_TOZERO)
#to compare my fn with the built in  one
display2Images(image5 , thresh1)

def SharpenAnImage(img):  # Sharpen
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv.filter2D(img, -1, kernel)
    display2Images(img, sharpened)

SharpenAnImage(image6)

def thirdTask(img):
    img1 = img.copy()
    blurred = cv.GaussianBlur(img, (5, 5), 0)  # low pass
    display2Images(img1, blurred)
    SharpenAnImage(img)
    canny_edges = cv.Canny(image=blurred, threshold1=100, threshold2=200)  # high pass
    display2Images(img1, canny_edges)

thirdTask(image5)

def fivethTask(img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    canny_edges = cv.Canny(image=img, threshold1=100, threshold2=200)  # high pass
    kernel = (2, 2)
    image = canny_edges.copy()
    for i in range(6):
        image = cv.dilate(image, kernel)
    display2Images(canny_edges, image)

fivethTask(image2)

def last1():
    img1 = cv.imread('boat cropped.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread('boat.jfif', cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()

last1()

def last2():
    img1 = cv.imread('boat cropped.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread('boat.jfif', cv.IMREAD_GRAYSCALE)  # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()

last2()

def last3():
    img1 = cv.imread('boat cropped.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread('boat.jfif', cv.IMREAD_GRAYSCALE)
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

last3()