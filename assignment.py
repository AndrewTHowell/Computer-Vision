#####################################################################
# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html
#####################################################################

# Section: Import modules

import cv2
import os
import numpy as np
import argparse
import sys
import math

# Section End

# Section: Set path to the video dataset

master_path_to_dataset = "D:/howel/Videos/Computer Vision Coursework"

# Section End

# Section: Constants

SEACHINGFOR = {"person": (114, 20, 34),
               "bicycle": (44, 85, 69),
               "car": (32, 33, 79),
               "motorbike": (52, 59, 41),
               "bus": (255, 35, 1),
               "truck": (248, 243, 53)}

# YOLO CNN object detection model
CONFIDENCETHRESHOLD = 0.5  # Confidence threshold
NMSTHRESHOLD = 0.4   # Non-maximum suppression threshold
INPWIDTH = 416       # WIDTH OF NETWORK'S INPUT IMAGE
INPHEIGHT = 416      # Height of network's input image

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5

# Section End

# Section: File handling

leftImagesPath = "left-images"     # edit this if needed
rightImagesPath = "right-images"   # edit this if needed

# resolve full directory location of data set for left / right images
fullPathLeftImages = os.path.join(master_path_to_dataset, leftImagesPath)
fullPathRightImages = os.path.join(master_path_to_dataset, rightImagesPath)

# get a list of the left image files and sort them (by timestamp in filename)
imageNameListL = sorted(os.listdir(fullPathLeftImages))

currentDirectoryPath = os.path.dirname(os.path.abspath(__file__))

class_file = os.path.join(currentDirectoryPath, "coco.names")
config_file = os.path.join(currentDirectoryPath, "yolov3.cfg")
weights_file = os.path.join(currentDirectoryPath, "yolov3.weights")

# Section End

# Section: Yolo functions


# dummy on trackbar callback function
def on_trackbar(val):
    return


# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in
def drawPred(image, class_name, distance, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = "{0}: {1:.2f}m".format(class_name, distance)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
                  (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression
def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores
    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)


# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Section End

# Section: Create stereoProcessor object

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval
maxDisparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, maxDisparity, 21)

# Section End

# Section: Create yolo object

# Load names of classes from file
classesFile = class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network using them
net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
output_layer_names = getOutputsNames(net)

# defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Section End

# Region: User input variables

cropDisparity = False  # display full or cropped disparity image
pausePlayback = False  # pause until key press after each image

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns
startTimestamp = ""  # set to timestamp to skip forward to

# Region End

# Section: 2D disparity to 3D


def project_disparity_to_3d(disparity, max_disparity, rgb=[]):

    points = []

    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    height, width = disparity.shape[:2]

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then we get reasonable scaling in X and Y output if we change
    # Z to Zmax in the lines X = ....; Y = ...; below

    # Zmax = ((f * B) / 2);

    for y in range(height):  # 0 - height is the y axis index
        for x in range(width):  # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y, x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = (f * B) / disparity[y, x]

                X = ((x - image_centre_w) * Z) / f
                Y = ((y - image_centre_h) * Z) / f

                # add to points
                if len(rgb) > 0:
                    points.append([X, Y, Z, rgb[y, x, 2], rgb[y, x, 1], rgb[y, x, 0]])
                else:
                    points.append([X, Y, Z])

    return points


# Section End

# Section: Iteration through image files

for imageNameL in imageNameListL:

    # Region: Skip to specific timestamp

    # Skip to timestamp file (if this is set)
    if ((len(startTimestamp) > 0) and not(startTimestamp in imageNameL)):
        continue
    elif ((len(startTimestamp) > 0) and (startTimestamp in imageNameL)):
        startTimestamp = ""

    # Region End

    # Derive path for right image from left image
    imageNameR = imageNameL.replace("_L", "_R")
    fullPathLeftImage = os.path.join(fullPathLeftImages, imageNameL)
    fullPathRightImage = os.path.join(fullPathRightImages, imageNameR)

    # check the file is a PNG file (left) and check a correspondoning right
    # image actually exists
    if ('.png' in imageNameL) and (os.path.isfile(fullPathRightImage)):

        # Region: Display left and right images

        # Read left and right images and display in windows
        imgL = cv2.imread(fullPathLeftImage, cv2.IMREAD_COLOR)
        cv2.imshow('Left image', imgL)
        imgR = cv2.imread(fullPathRightImage, cv2.IMREAD_COLOR)
        cv2.imshow('Right image', imgR)

        # Region End

        # Convert to grayscale (as the disparity matching works on grayscale)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Region: PREPROCESSING

        # Raise to the power, appears to improve subsequent disparity calculation
        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')

        # Region End

        # Compute disparity image (returned scaled by 16)
        disparity = stereoProcessor.compute(grayL, grayR)

        # Region: POSTPROCESSING

        # Filter out noise and speckles (adjust parameters as needed)
        dispNoiseFilter = 5  # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, maxDisparity - dispNoiseFilter)

        # Region End

        # Region: Prep for/Display Disparity Images

        # Scale the disparity to 8-bit for viewing
        _, disparity = cv2.threshold(disparity, 0, maxDisparity * 16,
                                     cv2.THRESH_TOZERO)
        disparityScaled = (disparity / 16.).astype(np.uint8)

        # If cropping wanted:
        if (cropDisparity):
            # Crop left part of disparity image where not seen by both cameras
            width = np.size(disparityScaled, 1)
            # Crop out the car bonnet
            disparityScaled = disparityScaled[0:390, 135:width]

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)
        scaledUpDisparity = (disparityScaled * (256. / maxDisparity)).astype(np.uint8)
        cv2.imshow("Disparity", scaledUpDisparity)

        # Now 3D
        points3D = project_disparity_to_3d(disparityScaled, maxDisparity)

        for i in range(10):
            print("points3D[i]: {0}".format(points3D[i]))

        # Region End

        # Region: YOLO

        windowName = "YOLO"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

        yoloImgL = imgL

        yoloWidth = yoloImgL.shape[1]
        yoloHeight = yoloImgL.shape[0]

        print("yoloImgL.shape: {0}".format(yoloImgL.shape))

        # start a timer (to see how long processing and display takes)
        start_t = cv2.getTickCount()

        # create a 4D tensor (OpenCV 'blob') from imgL (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(yoloImgL, 1/255, (yoloWidth, yoloHeight),
                                       [0, 0, 0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        classIDs, confidences, boxes = postprocess(yoloImgL, results,
                                                   CONFIDENCETHRESHOLD, NMSTHRESHOLD)

        # Region: Draw rects on image

        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            objectName = classes[classIDs[detected_object]]
            if objectName in SEACHINGFOR:
                box = boxes[detected_object]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]

                # Get distance of object
                # Get centre of object
                if (cropDisparity):
                    objectCentre = [top - (height//2), left + (width//2)]
                else:
                    objectCentre = [top - (height//2), left + (width//2)]
                print("objectCentre: {0}".format(objectCentre))

                """
                centreDisparity = scaledUpDisparity[objectCentre]
                print("centreDisparity: {0}".format(centreDisparity))

                objectDistance = (FOCALLENGTH
                                  * (BASELINEDISTANCE/centreDisparity))
                """
                objectDistance = 0.1

                drawPred(yoloImgL,
                         objectName,
                         objectDistance,
                         left, top, left + width, top + height,
                         SEACHINGFOR[objectName])

        # Region End

        # If cropping wanted:
        if (cropDisparity):
            # Crop left part of disparity image where not seen by both cameras
            # Crop out the car bonnet
            yoloImgL = yoloImgL[0:390, 135:yoloWidth]
            print("yoloImgL.shape: {0}".format(yoloImgL.shape))

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(yoloImgL, label,
                    (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255))

        # display image
        cv2.imshow(windowName, yoloImgL)
        cv2.resizeWindow(windowName, yoloImgL.shape[1], yoloImgL.shape[0])

        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        # Region End

        nearestObjectDistance = -1.0

        # Output filenames and nearest detected scene object
        print("{0}.png\n{1}.png : nearest detected scene object ({2:.1f}m)"
              .format(imageNameL, imageNameR, nearestObjectDistance))

        # exit - x
        # save - s
        # crop - c
        # pause - space
        # Region: User Input

        desiredFPS = 25
        keyDelay = 1000 / desiredFPS  # e.g: 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(max(2, keyDelay - int(math.ceil(stop_t))) * (not(pausePlayback))) & 0xFF

        # keyboard input for exit (as standard), save disparity and cropping
        if (key == ord('x')):       # exit
            break  # exit

        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparityScaled)
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)

        elif (key == ord('c')):     # crop
            cropDisparity = not(cropDisparity)

        elif (key == ord(' ')):     # pause (on next frame)
            pausePlayback = not(pausePlayback)

        # Region End

    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()

# Section End

# close all windows
cv2.destroyAllWindows()
