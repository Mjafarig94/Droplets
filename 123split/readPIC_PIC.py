import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from os import listdir
from os.path import isfile, join


def CountingCC():
    th, im_th = cv2.threshold(im_in, 0, 220, cv2.THRESH_BINARY_INV)
    cv2.imshow('nat',im_th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



mypath = '/Users/georgedamoulakis/PycharmProjects/Droplets/123split/splits2/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(' the files inside the folder are: ', onlyfiles)

images = np.empty(len(onlyfiles), dtype=object)
for n in range(1, len(onlyfiles)):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]), cv2.IMREAD_GRAYSCALE)
    im_in = images[n]
    cv2.imshow(f"Image {onlyfiles[n]}", im_in)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







