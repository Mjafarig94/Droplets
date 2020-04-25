import cv2 as cv2
import numpy as np

def unsharp_mask(image, kernel_size=(9, 9), sigma=2.0, amount=2.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

image = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Droplets/split.jpg', cv2.IMREAD_GRAYSCALE);
sharpened_image = unsharp_mask(image)


#first you do this

#for gamma in np.arange(0.0, 3.5, 0.1):
#	if gamma == 1:
#		continue
#	gamma = gamma if gamma > 0 else 0.1
#	adjusted = adjust_gamma(sharpened_image, gamma=gamma)
#	cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
#		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
#	cv2.imshow("Images", np.hstack([sharpened_image, adjusted]))
#	cv2.waitKey(0)


#second you do this
test = adjust_gamma(image, gamma=3.0)
cv2.imwrite('new input for drops.jpg', test)
cv2.imshow("Initial", image)
cv2.imshow("Final", test)
cv2.waitKey(0)
cv2.destroyAllWindows()