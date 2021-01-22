import cv2
import numpy as np

img = cv2.imread('alp.jpg', 0)

tmp = img.copy()

tmp = np.where(tmp > 200, 255, tmp)
tmp = np.where(tmp < 50, 255, tmp)
tmp = np.where(tmp < 200, 0, tmp)

ret, thresh = cv2.threshold(tmp, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    croped_img = img[y:y + h, x:x + w]
    croped_img = np.where(croped_img < 50, 0, 255)
    alpha = np.zeros_like(croped_img.shape, dtype=np.uint8)
    alpha = np.where(croped_img == 0, 255, 0)
    alpha = np.expand_dims(alpha, -1)
    croped_img = np.expand_dims(croped_img, -1)
    croped_img = np.repeat(croped_img, 3, -1)
    croped_img = np.concatenate([croped_img, alpha], 2)

    cv2.imwrite('AlpChars/' + str(i) + '.png', croped_img)
