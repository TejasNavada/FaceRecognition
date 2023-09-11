from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from dataset import TripletGenerator
from dataset import MapFunction
from model import SiameseModel
from matplotlib import pyplot as plt
import config
from tensorflow import keras
import tensorflow as tf
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src = 0).start()
time.sleep(2.0)

# create the data input pipeline for test dataset
print("[INFO] building the test generator...")
testTripletGenerator = TripletGenerator(
	datasetPath=config.TEST_DATASET)
print("[INFO] building the test `tf.data` dataset...")
testTfDataset = tf.data.Dataset.from_generator(
	generator=testTripletGenerator.get_next_identity,
	output_signature=(
		tf.TensorSpec(shape=(), dtype=tf.string),
		tf.TensorSpec(shape=(), dtype=tf.string),
	)
)

mapFunction = MapFunction(imageSize=config.IMAGE_SIZE)
print("[INFO] building the train and validation `tf.data` pipeline...")
testDs = (testTfDataset
    .map(mapFunction)
    .batch(len(testTripletGenerator.peopleNames))
    .prefetch(config.AUTO)
)

# load the siamese network from disk and build the siamese model
modelPath = config.MODEL_PATH
print(f"[INFO] loading the siamese network from {modelPath}...")
siameseNetwork = keras.models.load_model(filepath=modelPath)
siameseModel = SiameseModel(
    siameseNetwork=siameseNetwork,
    margin=0.5,
    lossTracker=keras.metrics.Mean(name="loss"),
)



# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels

    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()
    counter = 0
    # loop over the detections
    for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
        confidence = detections[0, 0, i, 2]
        
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence < args["confidence"]:
            continue
        counter+=1

        box = np.clip(detections[0, 0, i, 3:7], 0.0, 1.0)

        # compute the (x, y)-coordinates of the bounding
        # box for the object
        box = box * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # grab the face from the image, build the path to
        # the output face image, and write it to disk
        face = frame[startY:endY,startX:endX,:]

        #Convert img to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, config.IMAGE_SIZE)
        rgb_tensor = tf.convert_to_tensor(face, dtype=tf.float32)
        #Add dims to rgb_tensor
        rgb_tensor = tf.expand_dims(rgb_tensor , 0)
        print("positive: " + str(rgb_tensor.shape))
        # load the test dataqqq
        (positive, negative) = next(iter(testDs))
        print(positive.shape)
        (apDistance, anDistance) = siameseModel((rgb_tensor, positive, negative))

        minDistance = 0

        for person in range(1,len(testTripletGenerator.peopleNames)):
            if apDistance[person]<apDistance[minDistance]:
                minDistance=person

        # compute the (x, y)-coordinates of the bounding box for the
		# object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
		# probability
        text = testTripletGenerator.peopleNames[minDistance].partition('\\')[2] + ": {:0.2f}".format(apDistance[minDistance])
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    print(counter)    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

