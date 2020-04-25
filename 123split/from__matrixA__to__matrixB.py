import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from os import listdir
from os.path import isfile, join



############## I   N    F   O ##########################
# =======================================================
#Change#1: input image after Sharp and Gamma
#Change#2: better way to fix the image
#Change#3: new condition to LOGIC droplet or NOT
#Change#4: drawing circles only ON droplets
#Change#5: fix the "theoretical" radii of only the droplets
#Change#6: print the total number of droplets on final image

# to do:
#1st column: Average rate of growth of each droplet in 2 minutes
#2nd column: Average number of droplets in 2 minutes
#3rd column: Average  surface area of empty background in 2 minutes
#
# =======================================================


frames_number = 10

# here we have to built a matrix that takes the droplet number of each
# frame and store in a new matrix_A2 which has one column and rows
# as many as the frames_nuber
MatrixA2 = np.zeros((frames_number))



droplet_counter =


def matrixA_to_B():
#here we will build the MATRIX_A

    # 1st column: Average rate of growth of each droplet in 2 minutes
    # to find the average growth you need the area and the centroid of each droplet
    # DONE!!! 2nd column: Average number of droplets in 2 minutes
    # 3rd column: Average surface area of empty background in 2 minutes
    MatrixB = np.zeros((frames_number, 3))
    for row in range(frames_number):
        for column in range(0,3,1):
            if column == 0:
                MatrixB[row, column] = 1
            elif column == 1:
                MatrixB[row, column] = MatrixA2[row]
            elif column == 2:
                MatrixB[row, column] = 3
        column = column + 1
    row = row + 1


#save the new MatrixA to a csv file
    mypath = '/Users/georgedamoulakis/PycharmProjects/Droplets/matrices_A'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    df_M_A = pd.DataFrame(MatrixA)  # converting it to a pandas
    df_M_A.columns = [ 'Rate of Growth', 'Number of Droplets', 'Background Area']
    df_M_A.to_csv(f'MatrixA for image: {onlyfiles}.csv', index=False)  # save as csv

matrixA_to_B()