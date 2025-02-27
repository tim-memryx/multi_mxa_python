import os
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import queue
import time
import threading
from collections import deque
import argparse
from queue import Queue

import memryx
from memryx import AsyncAccl

class CartoonizerApp:
    def __init__(self, cam, mirror=True, src_is_cam=True):
        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.fps = int(cam.get(cv.CAP_PROP_FPS))
        self.capture_queue = Queue(maxsize=5)
        self.frame_times = deque(maxlen=30)
        self.mirror = mirror
        self.src_is_cam = src_is_cam

    # called at exit/EOF
    def _free(self, cap):
        cap.release()
        time.sleep(0.5) # a small delay allows a clean exit

    # Input Callback function for AsyncAccl
    def get_frame(self):

        # use an infinite loop and a src_is_cam check so
        # that we can "drop frames" on a slow host system
        # (when pre/post >> inference time)
        while True:
            ok, frame = self.cam.read()
            if not ok:
                print("EOF")
                return None
            if self.src_is_cam and self.capture_queue.full():
                # drop frame if queue is at capacity
                continue
            else:
                if self.mirror:
                    frame = cv.flip(frame, 1)

                # technically we don't need this queue since we're not drawing on
                # the original image, like we would for Object Detection, etc.
                #
                # but we'll leave this here for the sake of example anyway
                self.capture_queue.put(frame)

                return self.preprocess(frame)

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
        # technically we don't need this queue since we're not drawing on
        # the original image, like we would for Object Detection, etc.
        #
        # but we'll leave this here for the sake of example anyway
        self.capture_queue.get()
        
        # actual post processing
        img = self.postprocess(ofmaps[0], (self.input_width, self.input_height))
        self.display(img)
        return img

    # just displays in an OpenCV window
    def display(self, frame):
        # convert to OpenCV's BGR format
        img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("Cartoonizer demo", img)
        if cv.waitKey(1) == ord('q'):
            self._free(self.cam)
            exit(1)

# argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Cartoonizer demo")
    parser.add_argument('-d', '--dfp', default='../models/cartoonizer.dfp', help="Path to the DFP")
    parser.add_argument('-id', '--vid_cap_id', default=0, type=int, metavar="", help="Integer ID/index of the video capturing device to open and use")
    parser.add_argument('-x', '--mxa_id', default=0, type=int, help="MXA Device ID to run on (0,1,2,3)")
    args = parser.parse_args()
    return args

# camera open function
def setup_data(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")
    return cap




###############################################################################
def main():
    args = parse_args()

    video_src = args.vid_cap_id

    cap = setup_data(video_src)

    app = CartoonizerApp(cap, mirror=True, src_is_cam=True)

    # We set the "group_id" (more like "device_id") to the MXA we want to run on
    accl = AsyncAccl(args.dfp, group_id=args.mxa_id)

    # connect callbacks and start
    accl.connect_input(app.get_frame)
    accl.connect_output(app.process_model_output)

    # wait for EOF (user hits 'q')
    accl.wait()



if __name__ == '__main__':
    main()
