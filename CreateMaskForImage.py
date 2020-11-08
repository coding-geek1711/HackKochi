import cv2
import numpy as np
import shutil
import os


def readImage(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (720, 720))
    return image


# print(type(readImage("temp/20201107_141535.jpg")))


def createInitialMask(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
    # thresh = 170, mask = 255
    return mask


def erosionOfMask(mask):
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel)
    return eroded_mask


def findContours(eroded_mask):
    contours, heirarchy = cv2.findContours(
        eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    return contours, areas


def findIndividualGrains(contours, areas, originalImage):
    cropped_images = []

    for contour, area in zip(contours, areas):
        if area > 200 and area <= 5000:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_images.append(
                originalImage[y - 5: y + h + 5, x - 5: x + w + 5])

    if not os.path.exists("test_images"):
        os.mkdir("test_images")

    for image_number, img in enumerate(cropped_images):
        cv2.imwrite(f"test_images/{image_number}.jpg", img)

    return "saved to temp_images"


def drawBoundingRects(contours, listOfFlags, originalImage):
    print(originalImage)
    for contour, flag in zip(contours, listOfFlags):
        x, y, w, h = cv2.boundingRect(contour)
        start = (x, y)
        end = (x + w, y + h)
        print(flag)
        if flag == 'Average':
            cv2.rectangle(originalImage,
                          start, end, (0, 255, 255), 2)
            cv2.putText(originalImage, flag, start,
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        elif flag == 'Bad':
            cv2.rectangle(originalImage,
                          start, end, (0, 140, 255), 2)
            cv2.putText(originalImage, flag, start,
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        elif flag == 'Worse':
            cv2.rectangle(originalImage,
                          start, end, (0, 0, 255), 2)
            cv2.putText(originalImage, flag, start,
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        elif flag == 'Good':
            cv2.rectangle(originalImage,
                          start, end, (144, 238, 144), 2)
            cv2.putText(originalImage, flag, start,
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        elif flag == 'Excellent':
            cv2.rectangle(originalImage,
                          start, end, (0, 255, 0), 2)
            cv2.putText(originalImage, flag, start,
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    return originalImage


# ['Worse', 'Worse', 'Average', 'Worse', 'Average', 'Worse', 'Average', 'Worse', 'Worse', 'Worse', 'Average', 'Worse', 'Bad', 'Average', 'Worse', 'Worse', 'Average', 'Worse']
