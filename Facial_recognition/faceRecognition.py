import cv2
import numpy as np
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument("-i", "--image", required=True, help="path to input image")
argparser.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
argparser.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
argparser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(argparser.parse_args())

print("[INFO] Loading model....")

# load the model into opencv from disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

# load the input image into opencv
image = cv2.imread(args["image"])

#reshape image
(height, width) = image.shape[:2]

# resize image and convert to blob then normalize 
imageBlob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 1.0, (300,300), (104.0, 117.0, 123.0))


print("[INFO] computing object detections...")

# pass the blob to the network
net.setInput(imageBlob)

# obtain detections and predictions
detections = net.forward()


# loop over detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    #filter out weak detections by checking if confidence is lower than minimum confidence
    if confidence > args["confidence"]:
        # compute the (x, y) coordinates of the bounding box for the object
        boundingBox = detections[0, 0, i, 3:7] * np.array([width, height, width, height])

        (startX, startY, endX, endY) = boundingBox.astype("int")

        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        yAxis = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.putText(image, text, (startX, yAxis), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

#show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)