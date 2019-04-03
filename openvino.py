# from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2
import time
import sys
import argparse
import imutils
import time
import cv2
import os
import glob
from tqdm import tqdm
# 실시간 추적 모듈 sort 사용
from sort import *

cam_w = 320
cam_h = 240
image_size = 416

# cam에 맞는 size로 맞추기 위한 w,h
new_w = int(cam_w * min(image_size/cam_w, image_size/cam_h))
new_h = int(cam_h * min(image_size/cam_w, image_size/cam_h))

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
# net = IENetwork(model = xml_path,weights = bin_path)

# device (MYRIAD : NCS2)
# plugin = IEPlugin(device='MYRIAD')
# exec_net = plugin.load(net)

# cam on/cam setting
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

while(True) :
    # frame preprossing
    ret, frame = cam.read()
    resized_image = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

    # 128로 채운다
    canvas = np.full((image_size, image_size, 3), 128)
    canvas[(image_size - new_h) // 2:(image_size - new_h) // 2 + new_h, (image_size - new_w) // 2:(image_size - new_w) // 2 + new_w, :] = resized_image

    prepimg = canvas

    prepimg = resized_image[np.newaxis, :, :, :]  # Batch size axis add
    prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW

    start = time.time()

    cv2.imshow('image', prepimg)
    # inference
    # res = exec_net.infer({'inputs': prepimg})

    end = time.time()

    print('inference time : ', end - start)



    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()



