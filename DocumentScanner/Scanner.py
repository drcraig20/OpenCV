import sys
from imutils.perspective import four_point_transform
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

# find the contour of the edge image, keeping only the
# largest ones, and initialize the screen contour
contours = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if imutils.is_cv2() else contours[1]
contours = sorted(contours, key= cv2.contourArea, reverse= True)[:5]

# loop over the edge contours
for contour in contours:
	# approximate the contour
	peri = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenContour = approx

	
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenContour], -1, (0, 255, 0), 2)
cv2.imshow("outlined image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply a four point transformation to obtain a top-down view of the image
warped_image = four_point_transform(original_image, screenContour.reshape(4, 2) * ratio)

# convert the image to gray scale
warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

#  threshold the image
thresholdT = threshold_local(warped_image, 11, offset=10, method="gaussian")

warped_image = (warped_image > thresholdT).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(original_image, height = 650))
cv2.imshow("Scanned", imutils.resize(warped_image, height = 650))
cv2.waitKey(0)