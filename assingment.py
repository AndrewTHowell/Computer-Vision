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

# Section: File handling

leftImagesPath = "left-images"     # edit this if needed
rightImagesPath = "right-images"   # edit this if needed

# resolve full directory location of data set for left / right images
fullPathLeftImages = os.path.join(master_path_to_dataset, leftImagesPath)
fullPathRightImages = os.path.join(master_path_to_dataset, rightImagesPath)

# get a list of the left image files and sort them (by timestamp in filename)
imageNameListL = sorted(os.listdir(fullPathLeftImages))

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

# Region: User input variables

cropDisparity = False  # display full or cropped disparity image
pausePlayback = False  # pause until key press after each image

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns
startTimestamp = ""  # set to timestamp to skip forward to

# Region End

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
        cv2.imshow('left image', imgL)
        imgR = cv2.imread(fullPathRightImage, cv2.IMREAD_COLOR)
        cv2.imshow('right image', imgR)

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

        # Region: Prep for/Display Disparity Image

        # Scale the disparity to 8-bit for viewing
        _, disparity = cv2.threshold(disparity, 0, maxDisparity * 16, cv2.THRESH_TOZERO)
        disparityScaled = (disparity / 16.).astype(np.uint8)

        # If cropping wanted:
        if (cropDisparity):
            # Crop left part of disparity image where not seen by both cameras
            width = np.size(disparityScaled, 1)
            # Crop out the car bonnet
            disparityScaled = disparityScaled[0:390, 135:width]

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)
        cv2.imshow("disparity", (disparityScaled * (256. / maxDisparity)).astype(np.uint8))

        # Region End

        nearestObjectDistance = -1

        # Output filenames and nearest detected scene object
        print("{0}.png\n{1}.png : nearest detected scene object ({2:.1}m)"
              .format(imageNameL, imageNameR, nearestObjectDistance))

        # exit - x
        # save - s
        # crop - c
        # pause - space
        # Region: User Input

        desiredFPS = 25
        keyDelay = 1000 / desiredFPS  # e.g: 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(40 * (not(pausePlayback))) & 0xFF
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
