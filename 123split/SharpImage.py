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

def example():
    image = cv2.imread('/Users/georgedamoulakis/PycharmProjects/Droplets/123split/splits2/2.jpg', cv2.IMREAD_GRAYSCALE);
    sharpened_image = unsharp_mask(image)
    #cv2.imwrite('my-sharpened-image.jpg', sharpened_image)
    cv2.imshow("Initial", image)
    cv2.imshow("Final", sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

example()
