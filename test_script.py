import cv2
import numpy as np
# from CreateMaskForImage import drawBoundingRects

image = cv2.imread("UploadImages/test_image.jpg")
image = cv2.resize(image, (640, 640))

gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

_, mask = cv2.threshold(gray_scale, 170, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
eroded_mask = cv2.erode(mask, kernel)


contours, heirarchy = cv2.findContours(
    eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]

