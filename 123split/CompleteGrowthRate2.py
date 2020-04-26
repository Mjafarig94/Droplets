import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from os import listdir
from os.path import isfile, join


#------------------------
# otan sigkirineis ta existence kane sigrkisi SIMILAR kai oxi akribws to idio
#
#
# to problima pou prepei na ftia3w einai oti einai asta8is sta diafora
#mege8i - douleuei kala mono otan o 1os pinakas einai pio megalos h' isos
# me ton 2o
#----------------------

#insert the tables
M_frame = np.array((([1,2,65],
                     [13,6,50],
                     [100,100,100],
                     [5,5,200],
                     [60,60,400])), float)

M_frame15 = np.array((([13,6,80],
                       [1,2,10],
                       [80,80,600])),float)

#insert function to check and fix input matrices lenghts
def fix_matrices(Matrix1, Matrix2):
    Final_Matrix1 = Matrix1
    Final_Matrix2 = Matrix2
    if Matrix1.shape[0] == Matrix2.shape[0]:
        pass
    if Matrix1.shape[0] > Matrix2.shape[0]:
       while Matrix1.shape[0] > Matrix2.shape[0]:
           add_row = np.array([0, 0, 1])
           Matrix2 = np.vstack((Matrix2, add_row))
           Final_Matrix2 = np.array(Matrix2)
    if Matrix1.shape[0] < Matrix2.shape[0]:
        while Matrix1.shape[0] < Matrix2.shape[0]:
            add_row = np.array([0, 0, 1])
            Matrix1 = np.vstack((Matrix2, add_row))
            Final_Matrix1 = np.array(Matrix1)

    return Final_Matrix1, Final_Matrix2

# we building two matrices with the same number of rows
F_M_frame, F_M_frame15 = fix_matrices(M_frame, M_frame15)

#take only the two first columns - delete the third column
F_M_frame2 = np.delete(F_M_frame, 2, 1)
F_M_frame152 = np.delete(F_M_frame15, 2, 1)

# now check for existence of centroids from 1st to 2nd matrix
# function checks for existence
def check_for_existance(Matrix1, Matrix2):
    existance = np.empty(Matrix1.shape[0], dtype=object)
    existance2 = np.empty(Matrix1.shape[0], dtype=object)
    existance3 = np.empty(Matrix1.shape[0], dtype=object)
    for row in range(Matrix1.shape[0]):
        existance = np.array(np.where((Matrix2 == (Matrix1[row])).all(axis=1)))
        existance2[row] = existance
        if existance.size > 0:
            existance3[row] = 'exist'
        else:
            existance3[row] = 'no exist'
    return existance2, existance3

# numbers is a matrix with the new couples
numbers = check_for_existance(F_M_frame2, F_M_frame152)[0]
#exist_or is a matrix that shows existense or not
exist_or = check_for_existance(F_M_frame2, F_M_frame152)[1]

# here we are building the final matrix which gives us the
# correspondense which drops corresponds which other one
# in the second matrix
corresponding_matrix = np.empty(exist_or.shape, dtype=object)
for row in range(exist_or.shape[0]):
    if exist_or[row] == 'exist':
        corresponding_matrix[row] = (row, int(numbers[row]))
    else:
        corresponding_matrix[row] = (row, -1)
#print(corresponding_matrix)

#insert function almost same - meaning almost same centroids
def almost_same(number1, number2):
    NoChange = np.empty(1, dtype=object)
    if abs(number1 - number2) < 5:
        no_change = "same"
    else:
        no_change = 'different'
    return no_change

# the following for loops creating a matrix that shows what changes we have from frame to frame
# we have to compare the size difference for the same droplet
# to do that we will use the corresponding matrix
# the following for loops creating a matrix that shows what changes we have from frame to frame
diff_area = np.zeros(F_M_frame2.shape, dtype=float)
Changes = np.empty(F_M_frame2.shape, dtype=object)
for row in range(F_M_frame2.shape[0]):
    c1 = F_M_frame2[corresponding_matrix[row][0]]
    #print(c1)
    c2 = F_M_frame152[corresponding_matrix[row][1]]
    #print(c2)
    c1_3col = (F_M_frame[corresponding_matrix[row][0]])[2]
    c2_3col = (F_M_frame15[corresponding_matrix[row][1]])[2]
   # print(c2_3col - c1_3col)
    Changes[row,0] = almost_same(c1[0],c2[0])
    Changes[row, 1] = almost_same(c1[1], c2[1])
    if Changes[row,0] == 'same' and Changes[row, 1]=='same':
        diff_area[row] = (c2_3col - c1_3col)
    else:
        diff_area[row] = -1
#print(Changes)

Changes3 = np.empty(F_M_frame.shape, dtype=object)
for row in range(F_M_frame.shape[0]):
    for column in range(F_M_frame.shape[1]):
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
#print('LOGIC matrix')
#print(Changes3)

# now we have to fix the easy part of the algorithm when the droplet STAYS there
diff_area2 = np.zeros(F_M_frame.shape, dtype=float)
for row in range(F_M_frame.shape[0]):
    if diff_area[row, 0] == -1:
        diff_area2[row,0] = -1
    else:
        diff_area2[row, 0] = diff_area[row, 0] / F_M_frame[row][2]

#print('growth rates:')
#print(diff_area2)

#if there is a value of -1 (meaning problem) takes out the line
no_minusONE = (diff_area2 == -1).sum(1)
diff_area3 = diff_area2[no_minusONE == 0, :]
#print('SOLO growth rates:')
#print(diff_area3)

#now we have a clean matrix with real numbers and we can make the average
#print(f'The average growth of droplets is: ', np.average(diff_area3[:, 0]))

#now we have to deal with the areas that we dont have droplets
# numbers is a matrix with the new couples
numbers_GONE = check_for_existance(F_M_frame152, F_M_frame2)[0]
# exist_or is a matrix that shows existense or not
exist_or_GONE = check_for_existance(F_M_frame152, F_M_frame2)[1]

# print(exist_or_GONE )
Centr = np.zeros(F_M_frame15.shape, dtype=float)
for row in range(F_M_frame15.shape[0]):
    if exist_or_GONE[row] == 'no exist':
        if F_M_frame15[row][2] == 1:
            pass
        else:
            Centr[row, 0] = F_M_frame15[row][0]
            Centr[row, 1] = F_M_frame15[row][1]
            Centr[row, 2] = F_M_frame15[row][2]

print(Centr)

# if there is a value of 0 (meaning problem) takes out the line
no_minusZERO = (Centr == 0).sum(1)
Centr1 = Centr[no_minusZERO == 0, :]
#print(Centr1)

# now we have the centers that we want to look around for
# we will search the nearby area if there are centers in there


# check the values from Centr1 means the orphan droplets
# with centroids from the first frame that they have not passed through
possible_centr = np.zeros((F_M_frame15.shape[0],6), dtype=float)
for row in range(F_M_frame2.shape[0]):
    for rowCentr in range(Centr1.shape[0]):
        if Changes3[row][2] == 'droplet gone':
            if abs(Centr1[rowCentr][0] - F_M_frame2[row][0]) < 30 and \
                    abs(Centr1[rowCentr][1] - F_M_frame2[row][1])<30 and \
               abs(Centr1[rowCentr][2] - (F_M_frame[row][2]))>0:
                possible_centr[row][0] =  F_M_frame2[row][0]
                possible_centr[row][1] =  F_M_frame2[row][1]
                possible_centr[row][2] = F_M_frame[row][2]
                possible_centr[row][3] = Centr1[rowCentr][0]
                possible_centr[row][4] = Centr1[rowCentr][1]
                possible_centr[row][5] = Centr1[rowCentr][2]



print(possible_centr)