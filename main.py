from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2 as cv
import time
import argparse
import imutils
import time
import cv2
import os
import glob
from tqdm import tqdm
# 실시간 추적 모듈 sort 사용
from sort import *

xml_path = '/home/pi/workspace/IR/tiny-yolov3.xml'
bin_path = '/home/pi/workspace/IR/tiny-yolov3.bin'

LABELS = ("person", "bicycle", "car", "motorbike", "aeroplane",
          "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird",
          "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard","tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon",
          "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "book", "clock",
          "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

# network 생성
net = IENetwork(model = xml_path,weights = bin_path)

# device (MYRIAD : NCS2)
plugin = IEPlugin(device='MYRIAD')
exec_net = plugin.load(net)

start = time.time()

# image preprocessing
frame = cv.imread('test.jpeg')
resized_image = cv.resize(frame, (416, 416), interpolation = cv.INTER_CUBIC)
prepimg = resized_image[np.newaxis, :, :, :]     # Batch size axis add
prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW

end = time.time()

print('image process time : ',end - start)

start = time.time()

# inference
res = exec_net.infer({'inputs':prepimg})

end = time.time()

print('inference time : ',end-start)

cam = cv.VideoCapture(0)

'''
cam.set(cv2.CAP_PROP_FPS, vidfps)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
window_name = "USB Camera"
wait_key_time = 1
'''

print(cv.CAP_PROP_FPS)
print(cv.CAP_PROP_FRAME_WIDTH)
print(cv.CAP_PROP_FRAME_HEIGHT)

if cv.waitKey(1)&0xFF == ord('q'):
	sys.exit(0)




