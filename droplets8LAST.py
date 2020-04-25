import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


im_in = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Droplets/123split/1231.jpg', cv2.IMREAD_GRAYSCALE);
th, im_th = cv2.threshold(im_in, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

im_floodfill = im_th.copy()
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

kernel = np.ones((5,5),np.uint8)
im_floodfill = cv2.dilate(im_floodfill,kernel,iterations = 4)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
im_floodfill = cv2.erode(im_floodfill,kernel,iterations = 2)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(im_floodfill_inv, kernel, iterations=2)
dilation = cv2.dilate(erosion, kernel, iterations=2)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation)

y_min = []
y_max = []
x_min = []
x_max = []

for l in range(0, nlabels):
    y_min.append(0)
    y_max.append(0)
    x_min.append(0)
    x_max.append(0)

for l in range(0, nlabels):
    y_min_val = len(labels)
    y_max_val = 0
    x_min_val = len(labels[0])
    x_max_val = 0
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if (labels[i, j] == l):
                if i < y_min_val:
                    y_min_val = i
                if i > y_max_val:
                    y_max_val = i
                if j < x_min_val:
                    x_min_val = j
                if j > x_max_val:
                    x_max_val = j
    y_min[l] = y_min_val
    y_max[l] = y_max_val
    x_min[l] = x_min_val
    x_max[l] = x_max_val

to_black = []
for l in range(0, nlabels):
    d_ratio = (y_max[l] - y_min[l]) / (x_max[l] - x_min[l])
    if (d_ratio > 1.2) or (d_ratio < 0.8):
        to_black.append(l)

for b in to_black:
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if (labels[i, j] == b):
                labels[i, j] = 0

label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

a = np.hsplit(stats, 5)
horizontal = a[2]
vertical = a[3]
area = a[4]
b = np.hsplit(centroids, 2)
x_centr = b[0]
y_centr = b[1]

horizontalNEW = np.zeros(nlabels)
verticalNEW = np.zeros(nlabels)
TotalAreaNEW = np.zeros(nlabels)

NEW_dimensions = np.zeros((nlabels, 5))

for row in range(nlabels):
    for column in range(5):
        if column == 0:
            NEW_dimensions[row, column] = x_centr[row]
        elif column == 1:
            NEW_dimensions[row, column] = y_centr[row]
        elif column == 2:
            NEW_dimensions[row, column] = horizontalNEW[row]
        elif column == 3:
            NEW_dimensions[row, column] = verticalNEW[row]
        else:
            NEW_dimensions[row, column] = TotalAreaNEW[row]
    column = column + 1
row = row + 1

image1 = labeled_img
out = image1.copy()
grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
for row in range(2, nlabels, 1):
    for column in range(5):
        # to find the centers of each
        cv2.rectangle(out, (x_centr[row] - 3, y_centr[row] - 3), (x_centr[row] + 3, y_centr[row] + 3), (0, 0, 0))
        cv2.circle(out, (x_centr[row], y_centr[row]), int(((horizontalNEW[row] + verticalNEW[row]) * 0.25)), (255, 255, 255, 4))
        #r = (math.sqrt(TotalAreaNEW[row] * 0.31830988618) * 0.5)
        #cv2.circle(out, (x_centr[row], y_centr[row]), int(r), (255, 255, 0, 4))
    column = column + 1
row = row + 1

cv2.imshow("Final", out)

print(f'droplets = ', nlabels)
print(stats)
cv2.imshow("cc", labeled_img)
cv2.imshow("initial", im_in)
cv2.waitKey(0)

cv2.destroyAllWindows()

