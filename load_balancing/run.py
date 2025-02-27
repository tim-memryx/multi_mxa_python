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
apps = [None, None, None, None]


############################################
# Cartoonizer App                          #
############################################
class CartoonizerApp:
    def __init__(self, height:int, width:int):
        self.input_height = height
        self.input_width = width
        self.inputs_queue = Queue(maxsize=20)
        self.outputs_queue = Queue()

    def put_frame(self, frame):
        self.inputs_queue.put(frame)

    def get_output(self):
        return self.outputs_queue.get()

    # Input Callback function for AsyncAccl
    def get_frame(self):
        global running
        while running:
            # check every 5 seconds if we should exit
            try:
                frame = self.inputs_queue.get(timeout=5)
            except:
                continue
            else:
                return self.preprocess(frame)
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

# camera/video open function
def setup_src(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")
    return cap

# argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Combined Depth+Cartoonizer demo")
    parser.add_argument('-id', '--cam_id', default=0, type=int, metavar="", help="Integer ID/index of the video capturing device to open")
    parser.add_argument('-f', '--video_file', default="", type=str, help="Run on the given video file instead of a camera ID")
    args = parser.parse_args()
    return args



# continuously runs opencv camera capture and pushes to both apps
# NOTE: opencv and python threads like to misbehave, so that's why
#       we have only one cap and one imshow thread in this example
def capture_thread(cap):

    global running
    global apps

    # round-robin switching between running MXAs
    rr_cnt = 0

    while running:

        ok, frame = cap.read()
        if not ok:
            print("CAP EOF")
            running = False
            break

        # push to the currently selected obj
        apps[rr_cnt].put_frame(frame)
        if rr_cnt < 3:
            rr_cnt += 1
        else:
            rr_cnt = 0

    cap.release()



def main():
    
    global running
    global apps

    args = parse_args()

    if args.video_file != "":
        cap = setup_src(args.video_file)
        cap_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        cap_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    else:
        # open the camera
        cap = setup_src(args.cam_id)
        cap_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        cap_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    # create app objects
    apps = [CartoonizerApp(cap_h, cap_w),
            CartoonizerApp(cap_h, cap_w),
            CartoonizerApp(cap_h, cap_w),
            CartoonizerApp(cap_h, cap_w)]

    # start the camera capture thread
    tc = Thread(target=capture_thread, args=[cap], daemon=True)
    tc.start()

    # create accl objects
    accls = []
    for i in range(4):
        accls.append(AsyncAccl("../models/cartoonizer.dfp", group_id=i))

    # connect callbacks and start
    for i in range(4):
        accls[i].connect_input(apps[i].get_frame)
        accls[i].connect_output(apps[i].process_model_output)


    # same round-robin arbitrating
    rr_cnt = 0

    # use main as the "Display and Wait for 'q'" thread
    while running:
        
        # get frame
        frame = apps[rr_cnt].get_output()
        if rr_cnt < 3:
            rr_cnt += 1
        else:
            rr_cnt = 0

        # display and check 'q'
        cv.imshow("Load balancing demo", frame)
        if cv.waitKey(1) == ord('q'):
            # stop the cap thread
            running = False
            tc.join()

            time.sleep(5)

            # stop the accl objects
            for i in range(4):
                accls[i].stop()

            # exit
            break



if __name__ == '__main__':
    main()
