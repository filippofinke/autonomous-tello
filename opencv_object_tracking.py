from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from random import randint
from imageai.Detection import ObjectDetection
import os
from threading import Thread
import numpy as np
from tensorflow import logging
import sys
logging.set_verbosity(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

MARGIN_PERCENT = 20

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="Input file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

errors = []
trackers = []
bboxes = []
colors = [] 

vs = cv2.VideoCapture(args["video"])
if not vs.isOpened():
	sys.exit(0)

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(os.getcwd() , "yolo-tiny.h5"))
custom = detector.CustomObjects(person=True)
detector.loadModel(detection_speed="normal")
print("Model loaded!")

fps = FPS().start()
detect = 0
automatic = False
while True:
	detect += 1
	success, frame = vs.read()

	if not success:
		break


	frame = imutils.resize(frame, width=600)
	(H, W) = frame.shape[:2]

	marginX = round(W/100 * MARGIN_PERCENT)
	marginY = round(H/100 * MARGIN_PERCENT)
	endX = marginX + (W - marginX * 2)
	endY = marginY + (H - marginY * 2)
	if len(bboxes) > 0:
		for i, tracker in enumerate(trackers):
			success, box = tracker.update(frame)
			if success:
				errors[i] = 0
				color = colors[i]
				x = int(box[0])
				y = int(box[1])
				w = int(box[2])
				h = int(box[3])
				p1 = (x, y)
				p2 = (x + w, y + h)
				center = (x + round(w/2), y + round(h/2))
				if center[0] < marginX or center[0] > endX:
					color = (0,0,255)
				if center[1] < marginY or center[1] > endY:
					color = (0,0,255)	
				cv2.rectangle(frame, p1, p2, color, 2, 1)
				cv2.circle(frame, center, 5, color, 3)
			else:
				errors[i] += 1
				if errors[i] > 50:
					errors.pop(i)
					trackers.pop(i)
					bboxes.pop(i)
					colors.pop(i)
		for i, tracker in enumerate(trackers):
			cv2.putText(frame, "Tracker: " + ("Following" if errors[i] == 0 else "Lost"), (10, H - 70 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
		cv2.putText(frame, "Tracker: " + args["tracker"], (10, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	elif automatic:
		image, detections = detector.detectCustomObjectsFromImage(custom_objects=custom,input_type="array",input_image=frame,output_type="array")
		print(detections)
		for i, detection in enumerate(detections):
			bbox = detection["box_points"]
			bbox =  (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
			bboxes.append(bbox)
			colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
			tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
			tracker.init(frame, bbox)
			trackers.append(tracker)
			errors.append(0)
			cv2.imshow("DJI Tello AI", image)
			fps = FPS().start()
	fps.update()
	fps.stop()

	try:
		cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	except:
		cv2.putText(frame, "FPS: ERROR", (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	cv2.rectangle(frame, (marginX, marginY), (endX, endY), (0,0,255))
	cv2.putText(frame, "@filippofinke MODE " + ("Automatic" if automatic else "Manual"), (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	cv2.imshow("DJI Tello AI", frame)
	key = cv2.waitKey(1) & 0xFF


	if key == ord("s"):
		bbox = cv2.selectROI("DJI Tello AI", frame, fromCenter=False,
		showCrosshair=True)
		bboxes.append(bbox)
		colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		tracker.init(frame, bbox)
		trackers.append(tracker)
		errors.append(0)
		key = cv2.waitKey(1) & 0xFF
		fps = FPS().start()
	elif key == ord("w"):
		automatic = not automatic
	elif key == ord("r"):
		bboxes = []
		errors = []
		trackers = []
		colors = []
	elif key == ord("q"):
		break

vs.release()

cv2.destroyAllWindows()
