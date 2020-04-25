
#im_inG = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Droplets/count.jpg', cv2.IMREAD_GRAYSCALE).dtype
#im_inB = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Droplets/split.jpg.jpg', cv2.IMREAD_GRAYSCALE).shape

import numpy as np
import cv2
img = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Droplets/count.jpg', cv2.IMREAD_GRAYSCALE)
height, width, channels = img.shape
print(img.shape)