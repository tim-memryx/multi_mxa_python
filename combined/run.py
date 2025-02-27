import os
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import queue
import time
from threading import Thread
from collections import deque
import argparse
from queue import Queue

import memryx
from memryx import AsyncAccl


# variable to tell us to cleanly exit
running = True

# global vars instead of passing to Python threads,
# to prevent any buggyness with Threads
depth_app = None
cartoon_app = None



############################################
# Depth Estimation App                     #
############################################
class MidasApp:
    def __init__(self, height:int, width:int):
        self.input_height = height
        self.input_width = width
        self.inputs_queue = Queue()
        self.outputs_queue = Queue()

    def put_frame(self, frame):
        self.inputs_queue.put(frame)

    def get_output(self):
        return self.outputs_queue.get()

    # Input Callback function for AsyncAccl
    def get_frame(self):
        while running:
            # check every 5 seconds if we should exit
            try:
                frame = self.inputs_queue.get(timeout=5)
                return self.preprocess(frame)
            except:
                continue
        # if we get here, we should exit
        return None

    def preprocess(self, img):
        # get RGB to (0,1)
        arr = np.array(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (256, 256))).astype(np.float32)
        arr = arr/255.0

        # do the official MiDASv2 preprocessing steps
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        return arr

    def postprocess(self, frame, original_shape):
        # model output is an RGB image that needs to be scaled back up to [0,255]
        prediction = cv.resize(frame, original_shape)
        depth_min = prediction.min()
        depth_max = prediction.max()
        postprocessed_output = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        postprocessed_output = cv.applyColorMap(postprocessed_output, cv.COLORMAP_INFERNO)
        return postprocessed_output

    # Output Callback function for AsyncAccl
    def process_model_output(self, *ofmaps):
        img = self.postprocess(ofmaps[0], (self.input_width, self.input_height))
        self.outputs_queue.put(img)
        return img


############################################
# Cartoonizer App                          #
############################################
class CartoonizerApp:
    def __init__(self, height:int, width:int):
        self.input_height = height
        self.input_width = width
        self.inputs_queue = Queue()
        self.outputs_queue = Queue()

    def put_frame(self, frame):
        self.inputs_queue.put(frame)

    def get_output(self):
        return self.outputs_queue.get()

    # Input Callback function for AsyncAccl
    def get_frame(self):
        while running:
            # check every 5 seconds if we should exit
            try:
                frame = self.inputs_queue.get(timeout=5)
                return self.preprocess(frame)
            except:
                continue
        # if we get here, we should exit
        return None

    def preprocess(self, img):
        # this model expects inputs [-1, 1] and size [512,512,3]
        arr = np.array(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (512, 512))).astype(np.float32)
        arr = arr/127.5 - 1
        return arr

    def postprocess(self, frame, original_shape):
        # model output is an RGB image that needs to be scaled back up to [0,255]
        frame = (frame + 1) * 127.5
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = cv.resize(frame, original_shape)
        return frame

    # Output Callback function for AsyncAccl
    def process_model_output(self, *ofmaps):
        img = self.postprocess(ofmaps[0], (self.input_width, self.input_height))
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        self.outputs_queue.put(img)
        return img


############################################
# Shared Main Function                     #
############################################

# camera open function
def setup_cam(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")
    return cap

# argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Combined Depth+Cartoonizer demo")
    parser.add_argument('-id', '--cam_id', default=0, type=int, metavar="", help="Integer ID/index of the video capturing device to open")
    args = parser.parse_args()
    return args



# continuously runs opencv camera capture and pushes to both apps
# NOTE: opencv and python threads like to misbehave, so that's why
#       we have only one cap and one imshow thread in this example
def capture_thread(cap):

    global running
    global depth_app
    global cartoon_app

    while running:

        ok, frame = cap.read()
        if not ok:
            print("CAP EOF")
            break

        # mirror the image
        frame = cv.flip(frame, 1)

        # push to BOTH 
        depth_app.put_frame(frame)
        cartoon_app.put_frame(frame)

    cap.release()



def main():
    
    global running
    global depth_app
    global cartoon_app

    # HARDCODING the device IDs to 0 and 1 for the sake of example
    depth_mxa_id = 0
    cartoon_mxa_id = 1


    args = parse_args()

    # open the camera
    cap = setup_cam(args.cam_id)
    cap_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    # create app objects
    depth_app = MidasApp(cap_h, cap_w)
    cartoon_app = CartoonizerApp(cap_h, cap_w)

    # start the camera capture thread
    tc = Thread(target=capture_thread, args=[cap], daemon=True)
    tc.start()

    # create and connect accl objects
    depth_accl = AsyncAccl("../models/midas.dfp", group_id=depth_mxa_id)
    cartoon_accl = AsyncAccl("../models/cartoonizer.dfp", group_id=cartoon_mxa_id)

    # connect callbacks and start Depth
    depth_accl.connect_input(depth_app.get_frame)
    depth_accl.connect_output(depth_app.process_model_output)
    
    # connect callbacks and start Cartoonizer
    cartoon_accl.connect_input(cartoon_app.get_frame)
    cartoon_accl.connect_output(cartoon_app.process_model_output)


    # use main as the "Display and Wait for 'q'" thread
    while True:
        
        # get frames
        depth_out = depth_app.get_output()
        cartoon_out = cartoon_app.get_output()
        
        # concat them side-by-side
        img = cv.hconcat([depth_out, cartoon_out])

        # display and check 'q'
        cv.imshow("Combined demo", img)
        if cv.waitKey(1) == ord('q'):
            # stop the cap thread
            running = False
            tc.join()

            # stop the accl objects
            depth_accl.stop()
            cartoon_accl.stop()

            # exit
            break



if __name__ == '__main__':
    main()
