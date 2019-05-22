import ai2thor.controller
import keyboard
import numpy as np
import argparse
import cv2
import os
from PIL import Image
import matplotlib
import scipy
from scipy import misc
from scipy.misc import imread, imsave
import time
import datetime

#Starting
controller = ai2thor.controller.Controller()
controller.start()

#Choose scene
###print('List of scenes: (1-30),(201-230),(301-330),(401-430)')
###while True:
###	scene = input('Choose a scene you want to visit. ')
###	if (((int(scene) >= 1) and (int(scene) <= 30))
###			or ((int(scene) >= 201) and (int(scene) <= 230))
###			or ((int(scene) >= 301) and (int(scene) <= 330))
###			or ((int(scene) >= 401) and (int(scene) <= 430))):
###		break
###	else:
###		print ('Your scene number does not exist')
###room = 'FloorPlan' + scene
###controller.reset(room)
controller.reset('FloorPlan28')

#Initialization
controller.step(dict(action='Initialize', gridSize=0.1))

#controller.step(dict(action = 'InitialRandomSpawn', randomSeed = 0, forceVisible = False, maxNumRepeats = 20))
event = controller.step(dict(action='GetReachablePositions'))

# Numpy Array - shape (width, height, channels), channels are in RGB order
event.frame

# current metadata dictionary that includes the state of the scene
###event.metadata

# Some variables
last_pickupable_object_id = ''
last_receptacle_object_id = ''
objectName = ''
number = 1
distance = -1

def detect():
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3,
		help="threshold when applying non-maxima suppression")
	args = vars(ap.parse_args())

	# load the COCO class labels our YOLO model was trained on
	labelsPath = ("/home/quoctuan/darknet/coco.names")
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = ("/home/quoctuan/darknet/yolov3-tiny.weights")
	configPath = ("/home/quoctuan/darknet/yolov3-tiny.cfg")

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# load our input image and grab its spatial dimensions
	image = cv2.imread("/home/quoctuan/darknet/data/OD.jpg")
	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
	 
			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)
	 
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)


while True:
#Keyboard controller
	if keyboard.is_pressed('w'):
		event = controller.step(dict(action='MoveAhead'))

	if keyboard.is_pressed('s'):
		event = controller.step(dict(action='MoveBack'))

	if keyboard.is_pressed('a'):
		event = controller.step(dict(action='MoveLeft'))

	if keyboard.is_pressed('d'):
		event = controller.step(dict(action='MoveRight'))

	if keyboard.is_pressed('up arrow'):
		event = controller.step(dict(action='LookUp'))

	if keyboard.is_pressed('down arrow'):
		event = controller.step(dict(action='LookDown'))

	if keyboard.is_pressed('left arrow'):
		event = controller.step(dict(action='RotateLeft'))

	if keyboard.is_pressed('right arrow'):
		event = controller.step(dict(action='RotateRight'))

#Actions

	#Pickup an object
	if keyboard.is_pressed('q'):
		for o in event.metadata['objects']:
			if o['visible'] and o['pickupable']:
				if (distance == -1):
					distance = o['distance']
				else:
					if (distance > o['distance']):
						distance = o['distance']
		for o in event.metadata['objects']:
			if o['visible'] and o['pickupable']: #and (o['distance'] == distance):
				event = controller.step(dict(action='PickupObject', objectId=o['objectId']), raise_for_failure=False)
				last_pickupable_object_id = o['objectId']
				objectName = o['objectType']
				break
		if (event.metadata['lastActionSuccess'] == False):
			print(event.metadata['errorMessage'])
			distance = -1
		if (event.metadata['lastActionSuccess'] == True) and (event.metadata['lastAction'] == 'PickupObject'): 
			distance = -1
	
	#Put object on or into receptacle object
	if keyboard.is_pressed('r'):
		if (last_pickupable_object_id != ''):
			event = controller.step(dict(action='PutObject',
				receptacleObjectId=last_receptacle_object_id,
				objectId=last_pickupable_object_id), raise_for_failure=False)
		if (event.metadata['lastActionSuccess'] == False):
			print(event.metadata['errorMessage'])
		if (event.metadata['lastActionSuccess'] == True) and (event.metadata['lastAction'] == 'PutObject'): 
			objectName = ''

	#Open object
	if keyboard.is_pressed('u'):
		for o in event.metadata['objects']:
			if o['visible'] and o['openable'] and o['isopen']==False:
				if (distance == -1):
					distance = o['distance']
				else:
					if (distance > o['distance']):
						distance = o['distance']
		for o in event.metadata['objects']:
			if o['visible'] and o['openable'] and o['isopen']==False and (o['distance'] == distance):
				event = controller.step(dict(action='OpenObject', 
					objectId=o['objectId']), raise_for_failure=False)
				break
		if (event.metadata['lastActionSuccess'] == False):
			print(event.metadata['errorMessage'])
			distance = -1
		if (event.metadata['lastActionSuccess'] == True) and (event.metadata['lastAction'] == 'OpenObject'): 
			last_receptacle_object_id = o['objectId']
			objectName = o['objectType']
			distance = -1

	#Close object
	if keyboard.is_pressed('i'):
		for o in event.metadata['objects']:
			if o['visible'] and o['openable'] and o['isopen']==True:
				if (distance == -1):
					distance = o['distance']
				else:
					if (distance > o['distance']):
						distance = o['distance']
		for o in event.metadata['objects']:
			if o['visible'] and o['isopen']==True and (o['distance'] == distance) :
				event = controller.step(dict(action='CloseObject',
					objectId=o['objectId']), raise_for_failure=False)
				break
		if (event.metadata['lastActionSuccess'] == False):
			print(event.metadata['errorMessage'])
			distance = -1
		if (event.metadata['lastActionSuccess'] == True) and (event.metadata['lastAction'] == 'CloseObject'):
			last_receptacle_object_id = ''
			objectName = ''
			distance = -1
			
	#Drop object
	if keyboard.is_pressed('g'):
		event = controller.step(dict(action='DropHandObject'))
		if (event.metadata['lastActionSuccess'] == False):
			print(event.metadata['errorMessage'])
		if (event.metadata['lastActionSuccess'] == True) and (event.metadata['lastAction'] == 'DropHandObject'):
			objectName = ''
	
#Press those keys to get infomation
	if keyboard.is_pressed('o'):
		for o in event.metadata['objects']:
			if o['visible'] and o['pickupable']:
				print (o['objectId'])

	if keyboard.is_pressed('p'):
		print (event.metadata['agent'])

	if keyboard.is_pressed('f'):
		cv2.imwrite('OD.jpg', event.cv2img)
		os.rename("/home/quoctuan/Desktop/OD.jpg", "/home/quoctuan/darknet/data/OD.jpg")
		detect()

