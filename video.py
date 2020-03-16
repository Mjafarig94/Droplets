import cv2
import numpy as np

def HoughCir():
    image = cv2.imread('/Users/georgedamoulakis/PycharmProjects/a1/split.jpg')
    out = image.copy()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cir = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1.1, 100)
    cir = np.round(cir[0,:]).astype("int")
    for (x,y,r) in cir:
       cv2.circle(out, (x,y), r, (0,255,0,4))
       cv2.rectangle(out, (x-2, y-2), (x+2, y+2), (0,255,255))
    cv2.imshow("Final", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
