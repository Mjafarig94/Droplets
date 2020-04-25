import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from os import listdir
from os.path import isfile, join

#----------------------------------------
#
# will try to fix the growth rate problem
#
#
##----------------------------------------

def almost_same(number1, number2):
    NoChange = np.empty(1, dtype=object)
    if abs(number1 - number2) < 5:
        no_change = "same"
    else:
        no_change = 'different'
    return no_change


# lets take the matrices of two frames
#frame and frame15 which they have 15 secs time difference

# both of those frames will have a matrix with 3 columns and two rows
# the arrays have collumns like X,Y centroid and Area
M_frame = np.array((([1,2,65],
                     [13,6,50],
                     [10,10,10],
                     [15,15,400])), float)
M_frame15 = np.array((([1,2,80],
                       [13,6,80],
                       [10,10,10],
                       [15,15,600])),float)

# the following for loops creating a matrix that shows what changes we have from frame to frame
diff_area = np.zeros(M_frame.shape, dtype=float)
Changes = np.empty(M_frame.shape, dtype=object)
for row in range(M_frame.shape[0]):
    for column in range(M_frame.shape[1]):
        if column == 0:
            Changes[row, column] = almost_same(M_frame[row, column], M_frame15[row, column])
        if column == 1:
            Changes[row, column] = almost_same(M_frame[row, column], M_frame15[row, column])
        if column == 2:
            diff_area[row, column] = M_frame[row, column] - M_frame15[row, column]
    column = column + 1
row = row + 1


Changes3 = np.empty(M_frame.shape, dtype=object)
for row in range(M_frame.shape[0]):
    for column in range(M_frame.shape[1]):
        if column == 0:
            Changes3[row, column] = Changes[row, column]
        if column == 1:
            Changes3[row, column] = Changes[row, column]
        if column == 2:
            if Changes[row][0] == 'same' and Changes[row, 1] == 'same':
                Changes3[row, column] = 'ok'
            else:
                Changes3[row, column] = 'droplet gone'
    column = column + 1
row = row + 1
#print(Changes3)


# now we have to fix the easy part of the algorithm when the droplet STAYS there
diff_area2 = np.zeros(M_frame.shape, dtype=float)
for row in range(M_frame.shape[0]):
    for column in range(M_frame.shape[1]):
        if Changes3[row][2] == 'ok':
            diff_area2[row, column] = (diff_area[row, column] / M_frame[row, column])
        else:
            diff_area2[row, column] = -1
    column = column + 1
row = row + 1


print(f'The average growth of droplets is: ', np.average(diff_area2[:, 2]))
