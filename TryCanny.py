import cv2
import numpy as np

img = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Droplets/split.jpg',0)
edges = cv2.Canny(img,100,100,apertureSize = 3)
result = np.hstack((img,edges))

#kernel = np.ones((3,3),np.uint8)
#erosion = cv2.erode(edges, kernel, iterations=1)
#dilation = cv2.dilate(erosion,kernel,iterations=0)
#result1 = np.hstack((erosion,dilation))

def CC(img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    return labeled_img, nlabels, labels, stats, centroids

#count droplets
components, nlabels, labels, stats, centroids = CC(edges)
print(f' There are ', nlabels, ' droplets')
#print(f' with the following labels: ', labels)


print('Original Dimensions : ',img.shape)
scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(edges, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',resized.shape)

cv2.imshow('image',result)
#cv2.imshow('erosion/ dilation',result1)
cv2.imshow('final',components)
cv2.waitKey()
cv2.destroyAllWindows()


HoughCir()