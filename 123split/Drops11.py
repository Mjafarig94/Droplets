import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


############## I   N    F   O ##########################
# =======================================================
#
#1st column: Average rate of growth of each droplet in 2 minutes
#2nd column: Average number of droplets in 2 minutes
#3rd column: Average  surface area of empty background in 2 minutes
#
# =======================================================

# Read image
im_in = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Droplets/123split/splits2/2.jpg', cv2.IMREAD_GRAYSCALE);


def CountingCC():
    # Threshold, Set values equal to or above 220 to 0, Set values below 220 to 255.
    th, im_th = cv2.threshold(im_in, 40, 255, cv2.THRESH_BINARY_INV);

    def CC(img):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue == 0] = 0
        return labeled_img, nlabels, labels, stats, centroids


    kernel = np.ones((3, 3), np.uint8)
    erosion2 = cv2.erode(im_th, kernel, iterations=3)
    dilation2 = cv2.dilate(erosion2, kernel, iterations=3)
    blur4 = cv2.blur(dilation2, (5, 5))
    erosion = cv2.erode(blur4, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    components, nlabels, labels, stats, centroids = CC(dilation2)

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


#Logic check if something is DROPLET or NOT
    d=0
    Not_Droplet = np.empty(nlabels, dtype=object)
    for i in range(nlabels):
        d = ((horizontal[i] + vertical[i]) / 2)
        d1 = 0.785*d*d
        if abs(area[i] - (d1) ) > 200:
            print(f'the #', i, ' object is not a droplet')
            print(f'the area is', area[i], 'the d1 is ', d1)
            Not_Droplet[i] = "NOT a droplet"
        else:
            Not_Droplet[i] = "ok"


    NEW_dimensions = np.zeros((29, 6))

    for row in range(nlabels):
        for column in range(8):
            if column == 0:
                NEW_dimensions[row, column] = (row + 1)
            elif column == 1:
                NEW_dimensions[row, column] = x_centr[row]
            elif column == 2:
                NEW_dimensions[row, column] = y_centr[row]
            elif column == 3:
                NEW_dimensions[row, column] = horizontal[row] + 20
            elif column == 4:
                NEW_dimensions[row, column] = vertical[row] + 20
            elif column == 5:
                NEW_dimensions[row, column] = ((NEW_dimensions[row][3]) + (NEW_dimensions[row][4])) * 3.14 * 0.25 * ((NEW_dimensions[row][3]) + (NEW_dimensions[row][4]))
        column = column + 1
    row = row + 1
    plt.show()

    image = components
    out = image.copy()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for row in range(1, nlabels, 1):
        for column in range(5):
            cv2.rectangle(out, (x_centr[row] - 3, y_centr[row] - 3), (x_centr[row] + 3, y_centr[row] + 3),
                          (0, 0, 0))
            r = (math.sqrt(NEW_dimensions[row][5] * 0.31830988618) * 0.5)
            cv2.circle(out, (x_centr[row], y_centr[row]), int(r), (255, 255, 0, 4))
            cv2.putText(out, ('%d' % (row + 1)), (x_centr[row], y_centr[row]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
        column = column + 1
    row = row + 1

    cv2.imshow("Initial", im_in)
    cv2.imshow("Final", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    my_df_1 = pd.DataFrame(NEW_dimensions)  # converting it to a pandas
    my_df_2 = pd.DataFrame(Not_Droplet)  # converting it to a pandas
    frames = [my_df_1, my_df_2]
    my_df = pd.concat(frames, axis=1, sort=False)
    my_df.columns = ['Droplet #', 'X-Centroid', 'Y-Centroid', 'Horizontal', 'Vertical', 'Area',
                     'Droplet/No Droplet']
    my_df.to_csv('dimensions.csv', index=False)  # save as csv


CountingCC()