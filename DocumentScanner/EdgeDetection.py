import numpy as np
from skimage.filters import threshold_local
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
original_image = image.copy()
image = imutils.resize(image, height=500)

# convert the image to grayscale, make it blurry and find edges
# grayout image
grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur out image
smoothened_grayed_image = cv2.GaussianBlur(grayed_image, (5, 5), 0)
# get edges
edged_image = cv2.Canny(smoothened_grayed_image, 75, 200)

print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
