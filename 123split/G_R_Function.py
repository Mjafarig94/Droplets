import numpy as np

#------------------------
# otan sigkirineis ta existence kane sigrkisi SIMILAR kai oxi akribws to idio
#
#
# to problima pou prepei na ftia3w einai oti einai asta8is sta diafora
#mege8i - douleuei kala mono otan o 1os pinakas einai pio megalos h' isos
# me ton 2o
#----------------------

#insert the tables
first_matrix = np.array((([1,2,65],
                     [13,6,50],
                     [100,100,500],
                     [50,50,55],
                     [5,5,200],
                     [60,60,400])), float)

second_matrix = np.array((([13,6,51],
                       [1,2,66],
                       [80,80,550],
                       [1000,1000,100],
                       [90,90,500])),float)

def GrowthRate(M_frame, M_frame15):
    # insert function to check and fix input matrices lenghts
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

    # take only the two first columns - delete the third column
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
    # exist_or is a matrix that shows existense or not
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

    # print(corresponding_matrix)

    # insert function almost same - meaning almost same centroids
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
        # print(c1)
        c2 = F_M_frame152[corresponding_matrix[row][1]]
        # print(c2)
        c1_3col = (F_M_frame[corresponding_matrix[row][0]])[2]
        c2_3col = (F_M_frame15[corresponding_matrix[row][1]])[2]
        # print(c2_3col - c1_3col)
        Changes[row, 0] = almost_same(c1[0], c2[0])
        Changes[row, 1] = almost_same(c1[1], c2[1])
        if Changes[row, 0] == 'same' and Changes[row, 1] == 'same':
            diff_area[row] = (c2_3col - c1_3col)
        else:
            diff_area[row] = -1
    # print(Changes)

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
    # print('LOGIC matrix')
    # print(Changes3)

    # now we have to fix the easy part of the algorithm when the droplet STAYS there
    diff_area2 = np.zeros(F_M_frame.shape, dtype=float)
    for row in range(F_M_frame.shape[0]):
        if diff_area[row, 0] == -1:
            diff_area2[row, 0] = -1
        else:
            diff_area2[row, 0] = diff_area[row, 0] / F_M_frame[row][2]

    # print('growth rates:')
    # print(diff_area2)

    # if there is a value of -1 (meaning problem) takes out the line
    no_minusONE = (diff_area2 == -1).sum(1)
    diff_area3 = diff_area2[no_minusONE == 0, :]
    #print('----------------------------------------------------------------')
    #print('This matrix shows as the growth rate of the ')
    #print('droplets that exist in both frames :')
    #print(diff_area3[:, 0])
    #print('----------------------------------------------------------------')

    # now we have a clean matrix with real numbers and we can make the average
    exist_grow_rate = np.average(diff_area3[:, 0])
    #print('This is the average growth rate  ')
    #print(f' of the existing droplets:  ', round(exist_grow_rate, 3))
    #print('----------------------------------------------------------------')

    # now we have to deal with the areas that we dont have droplets
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

    # print(Centr)

    # if there is a value of 0 (meaning problem) takes out the line
    no_minusZERO = (Centr == 0).sum(1)
    Centr1 = Centr[no_minusZERO == 0, :]
    #print(f'The new droplets at the second frame are:{Centr1}')
    #print('----------------------------------------------------------------')

    # now we have the centers that we want to look around for
    # we will search the nearby area if there are centers in there

    # check the values from Centr1 means the orphan droplets
    # with centroids from the first frame that they have not passed through
    possible_centr = np.zeros((F_M_frame15.shape[0], 6), dtype=float)
    for row in range(F_M_frame2.shape[0]):
        for rowCentr in range(Centr1.shape[0]):
            if Changes3[row][2] == 'droplet gone':
                if abs(Centr1[rowCentr][0] - F_M_frame2[row][0]) < 30 and \
                        abs(Centr1[rowCentr][1] - F_M_frame2[row][1]) < 30 and \
                        Centr1[rowCentr][2] - (F_M_frame[row][2]) > 0:
                    possible_centr[row][0] = F_M_frame2[row][0]
                    possible_centr[row][1] = F_M_frame2[row][1]
                    possible_centr[row][2] = F_M_frame[row][2]
                    possible_centr[row][3] = Centr1[rowCentr][0]
                    possible_centr[row][4] = Centr1[rowCentr][1]
                    possible_centr[row][5] = Centr1[rowCentr][2]

    #print('This matrix shows the possible coelecence events:')
    #print('       First frame   -  Second frame ')
    #print(possible_centr)
    #print('----------------------------------------------------------------')

    possible_centr_3col = np.zeros((F_M_frame15.shape[0], 3), dtype=float)
    for row in range(possible_centr.shape[0]):
        possible_centr_3col[row][0] = possible_centr[row][3]
        possible_centr_3col[row][1] = possible_centr[row][4]
        possible_centr_3col[row][2] = possible_centr[row][5]

    # now lets find the rate growth
    growth_rate_collision = np.zeros((possible_centr.shape[0], 1), dtype=float)
    for row in range(possible_centr.shape[0]):
        if possible_centr[row][0] == 0:
            pass
        else:
            growth_rate_collision[row] = ((possible_centr[row][5] - possible_centr[row][2]) / possible_centr[row][2])

    # print('the growth rate for collisiions is the following:')
    # print(growth_rate_collision)
    # print('----------------------------------------------------------------')

    # if there is a value of 0 (meaning problem) takes out the line
    no_minusZERO_col = (growth_rate_collision == 0).sum(1)
    growth_rate_collision_clean = growth_rate_collision[no_minusZERO_col == 0, :]
    #print(f'the clean matrix for growth rate for collisions is :')
    #print(growth_rate_collision_clean)
    #print('----------------------------------------------------------------')

    # now we have a clean matrix with real numbers and we can make the average
    collision_grow_rate = np.average(growth_rate_collision_clean)
    #print('This is the average growth rate  ')
    #print(f' of the collision droplets:  ', round(collision_grow_rate, 3))
    #print('----------------------------------------------------------------')
    # numbers is a matrix with the new couples
    numbers_new_baby = check_for_existance(Centr1, possible_centr_3col)[0]
    # exist_or is a matrix that shows existense or not
    exist_or_new_baby = check_for_existance(Centr1, possible_centr_3col)[1]
    new_baby = np.zeros((Centr1.shape[0], 3), dtype=float)
    for row in range(Centr1.shape[0]):
        if exist_or_new_baby[row] == 'no exist':
            new_baby[row] = Centr1[row]

    # print('This matrix shows the birth of new droplets:')
    # print(new_baby)
    # print('----------------------------------------------------------------')

    # if there is a value of 0 (meaning problem) takes out the line
    no_minusZERO_baby = (new_baby == 0).sum(1)
    growth_rate_baby_clean = new_baby[no_minusZERO_baby == 0, :]
    #print(f'the clean matrix for growth rate for collisions is :')
    #print(growth_rate_baby_clean)
    #print('----------------------------------------------------------------')

    # now lets collect all the previous data to give the final answer about the
    # growth rate of the frame to frame

    # first lets count the droplets
    def percent_of_droplet_style():
        nu_total_droplets = M_frame15.shape[0]
        nu_simple_grow_droplets = (diff_area3[:, 0]).shape[0]
        nu_collision_grow_droplets = growth_rate_collision_clean.shape[0]
        nu_baby_droplets = growth_rate_baby_clean.shape[0]

        weight_simple = (nu_simple_grow_droplets / nu_total_droplets)
        weight_col = (nu_collision_grow_droplets / nu_total_droplets)
        weight_baby = (nu_collision_grow_droplets / nu_total_droplets)

        total_Growth_Rate = round(
            (weight_simple * exist_grow_rate) + (weight_col * collision_grow_rate) + (weight_baby * 0.5), 3)
        return total_Growth_Rate

    #print(f' The total Growth Rate of this frame is : ', percent_of_droplet_style())
    #print('----------------------------------------------------------------')
    #print('----------------------------------------------------------------')

    return percent_of_droplet_style()

G_R_matrix = GrowthRate(first_matrix, second_matrix)
print(f'The growth rate of this frame to frame pass is :', G_R_matrix)
