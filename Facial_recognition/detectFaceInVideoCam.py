import cv2
from imutils.video import VideoStream
import time
import imutils
import argparse
import numpy as np

argparser = argparse.ArgumentParser()

argparser.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
argparser.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
argparser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(argparser.parse_args())

# load model from disk to OpenCV
print("[INFO] loading model...")

# load the model into opencv neural network
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# start live video stream and allow camera stream to start up
print("[INFO] Started video streaming")
videoStreaming = VideoStream(src=0).start()
time.sleep(2.0)

# loop through the video frames and detect faces
while True:
    # read threaded frames from the video stream
    frame = videoStreaming.read()

    # resize the frame to a minimum of width of 400px
    frame = imutils.resize(frame, width=400)

    # get frame width and heigth
    (height, width) = frame.shape[:2]

    # convert frame to blob
    blobFrame = cv2.dnn.blobFromImage(cv2.resize(frame,(300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain detections and preditions
    net.setInput(blobFrame)
    detections = net.forward()

    # looping over results from detections
    for i in range(0, detections.shape[2]):
        # extract the confidence i.e, probability associated with the predictions
        confidence = detections[0, 0, i, 2]

        # filter out confidence level that are lower the the default confidence level
        defaultConfidence = args["confidence"]

        # if confidence level is greater then continue execution
        if confidence < defaultConfidence:
            continue

        # compute the bounding box (x, y) coordinates of the detected object
        boundingBox = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (startX, startY, endX, endY) = boundingBox.astype("int")

        # draw the boundingbox of the face along with the probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0,255), 2)

        # put the text on the required position
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # show the output frame
        cv2.imshow("Video Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed then break from the loop
        if key == ord("q"):
            break


cv2.destroyAllWindows()
videoStreaming.stop()
        
