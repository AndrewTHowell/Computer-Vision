currentPath = ("D://howel//OneDrive - Durham University//Degree//Year 3//SSA//"
               "Computer Vision//Coursework//Computer-Vision//")
videoPath = "D://howel//Videos//Computer Vision Coursework"

keep_processing = True

# parse command line arguments for camera ID or video file, and YOLO files
parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument("-fs", "--fullscreen", action='store_true', help="run in full screen mode")
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
parser.add_argument("-cl", "--class_file", type=str, help="list of classes", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str, help="network config", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str, help="network weights", default='yolov3.weights')

args = parser.parse_args()

# define video capture object

try:
    # to use a non-buffered camera stream (via a separate thread)

    if not(args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture() # not needed for video files

except:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

################################################################################

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Load names of classes from file

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network using them

net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights_file)
output_layer_names = getOutputsNames(net)

 # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

################################################################################

# define display window name + trackbar

windowName = 'YOLOv3 object detection: ' + args.weights_file
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, windowName , 0, 100, on_trackbar)

################################################################################

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    while (keep_processing):

        # start a timer (to see how long processing and display takes)
        start_t = cv2.getTickCount()

        # if camera /video file successfully open then read frame
        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly
            if (ret == 0):
                keep_processing = False
                continue

            # rescale if specified
            if (args.rescale != 1.0):
                frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        confThreshold = cv2.getTrackbarPos(trackbarName,windowName) / 100
        classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)

        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50))

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image
        cv2.imshow(windowName,frame)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN & args.fullscreen)

                                # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        # start the event loop + detect specific key strokes
        # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        # if user presses "x" then exit  / press "f" for fullscreen display
        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            args.fullscreen = not(args.fullscreen)

    # close all windows
    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

################################################################################
