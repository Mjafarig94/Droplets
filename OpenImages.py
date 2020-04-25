from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import matplotlib.pyplot as plt

X = []
Y = []

def CountingCC():
# Threshold, Set values equal to or above 220 to 0, Set values below 220 to 255.
    th, im_th = cv2.threshold(im_in, 0, 220, cv2.THRESH_BINARY_INV)

    def CC (img):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue==0] = 0
        return labeled_img, nlabels, labels, stats, centroids

#cleaning up the image get it ready to count
    blur = cv2.blur(im_th,(9,9))
    blur2 = cv2.blur(blur,(3,3))
    blur3 = cv2.blur(blur2,(1,1))
    kernel = np.ones((10,10),np.uint8)
    erosion2 = cv2.erode(blur3, kernel, iterations=4)
    dilation2 = cv2.dilate(erosion2,kernel,iterations=3)
    blur4 = cv2.blur(dilation2,(5,5))
    erosion = cv2.erode(blur4, kernel, iterations=1)
    dilation = cv2.dilate(erosion,kernel,iterations=0)

    #count droplets
    components, nlabels, labels, stats, centroids = CC(dilation)
    #print(f' There are ', nlabels, ' droplets')
    #print(f' with the following labels: ', labels)
    Y.append(nlabels)


mypath='/Users/georgedamoulakis/PycharmProjects/Droplets/123split'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]

images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread(join(mypath,onlyfiles[n]), cv2.IMREAD_GRAYSCALE)
  im_in = images[n]
  CountingCC()
  X.append(n)


# plotting the points
plt.plot(X,Y)
# naming the x axis
plt.xlabel('Frame')
# naming the y axis
plt.ylabel('Number of Droplets')
# giving a title to my graph
plt.title('Droplets v Frame')
# function to show the plot
plt.show()

plt.scatter(X, Y, label= "droplets", color= "red", marker= "*", s=30)
# naming the x axis
plt.xlabel('Frame')
# naming the y axis
plt.ylabel('Number of Droplets')
# giving a title to my graph
plt.title('Droplets v Frame')
# function to show the plot
plt.show()

