import cv2
from CreateMaskForImage import drawBoundingRects

img = cv2.imread("temp/20201107_135456.jpg")
x, y, w, h = 511, 583, 11, 20

cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv2.imshow("IMAGE", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
