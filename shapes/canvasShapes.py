import numpy as np
import cv2

# create and image canvas with blank pixels
canvas = np.zeros((300, 300, 3), dtype="uint8")

# tuple to represent green color in BGR
green = (0, 255, 0)

# tuple to represent red color in BGR
red = (0, 0, 255)

# draw red line 2px thick from top right corner to bottom left corner
cv2.line(canvas, (300, 0), (0, 300), red, 3)

# draw green line 1px thick from top left corner to bottom right corner
cv2.line(canvas, (0, 0), (300, 300), green)

# draw a green rectangle in the specified location 
cv2.rectangle(canvas, (10, 10), (60, 60), green, -1)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# reinitiate a plain canvas 
canvas = np.zeros((500, 500, 3), dtype="uint8")

# get the center of the image
(centerX, centerY) = (canvas.shape[1] // 2 , canvas.shape[0] // 2)

# tuple to represent white color in BGR
white = (255, 255, 255)

# draw multiple white circle with increasing radius in the image
for radius in range(0, 500, 25):
    cv2.circle(canvas, (centerX, centerY), radius, white, 2)

cv2.imshow("Circles", canvas)
cv2.waitKey(0)

# reinitiate a plain canvas 
canvas = np.zeros((500, 500, 3), dtype="uint8")

for i in range(0, 25):
    # generate random numbers for radius
    radius = np.random.randint(low=5, high=200)

    # generate random colors
    color = np.random.randint(low=0, high=256, size=(3,)).tolist()

    # generate random location
    points = np.random.randint(low=0, high=500, size=(2,))

    cv2.circle(canvas, tuple(points), radius, color, -1)

cv2.imshow("Random color cirle", canvas)
cv2.waitKey(0)

cv2.imwrite("Random colors.jpeg", canvas)