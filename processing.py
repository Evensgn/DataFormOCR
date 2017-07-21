import cv2
import numpy as np

GRAY_SCALE_RANGE = 255
NUM_ROW = 28
NUM_COLUMN = 28

img = cv2.imread("photo.jpg")
# cv2.imshow('img', img)
# cv2.waitKey(0)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img_gray', img_gray)
# cv2.waitKey(0)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
# cv2.imshow('img_gray', img_gray)
# cv2.waitKey(0)
ret, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('img_thresh', img_thresh)
# cv2.waitKey(0)
im2, ctrs, hier = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

cv2.imshow('img_thresh', img_thresh)
cv2.waitKey(0)

img_digits = []
for rect in rects:
	cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 255), 3)
	img_digit = img_thresh[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
	# cv2.imshow('img_digit', img_digit)
	# cv2.waitKey(0)
	img_digits.append(img_digit)

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * NUM_ROW * skew], [0, 1, 0]]) # require that NUM_ROW == NUM_COLUMN
    img = cv2.warpAffine(img, M, (NUM_ROW, NUM_ROW), flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

images = []
is_decimal_point = []
for img_digit in img_digits:
	square_len = max(img_digit.shape[0], img_digit.shape[1]) 
	bdr_len0 = int((square_len - img_digit.shape[0]) / 2)
	bdr_len1 = int((square_len - img_digit.shape[1]) / 2)
	img_digit = cv2.copyMakeBorder(img_digit, bdr_len0, bdr_len0, bdr_len1, bdr_len1, cv2.BORDER_CONSTANT, value = 0)
	if np.mean(img_digit) > 145:
		is_decimal_point.append(True)
	else:
		is_decimal_point.append(False)
	margin = int(square_len * 0.175)
	img_margin = cv2.copyMakeBorder(img_digit, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value = 0)
	img_margin = cv2.resize(img_margin, (NUM_ROW, NUM_COLUMN), interpolation = cv2.INTER_AREA)
	img_margin = deskew(img_margin)	
	images.append(img_margin)

import classifier

digits = classifier.classify(images)
for i in range(len(digits)):
	if is_decimal_point[i]:
		digits[i] = 10
print(digits)

for i in range(len(digits)):
	if digits[i] < 10:
		cv2.putText(img, str(digits[i]), (rects[i][0], rects[i][1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
	else:
		cv2.putText(img, '.', (rects[i][0], rects[i][1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
cv2.imshow('processed img', img)
cv2.waitKey(0)
