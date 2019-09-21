try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin

import logging
from logging import handlers
import cv2
import imutils
import time
import os
import schedule
from apscheduler.schedulers.background import BackgroundScheduler

from inference import *

# model setting

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

classes = 80
coords = 4
num = 3

image_size = 416
cam_w = 416
cam_h = 416

new_w = int(cam_w * min(image_size/cam_w, image_size/cam_h))
new_h = int(cam_h * min(image_size/cam_w, image_size/cam_h))

xml_path = "./IR/yolo_v3.xml" #<--- MYRIAD
bin_path = os.path.splitext(xml_path)[0] + ".bin"
img_path = './test.jpg'

# color setting
label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1

drawing = False

# logger
logger = logging.getLogger("")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

stream_hander = logging.StreamHandler()
stream_hander.setFormatter(formatter)
logger.addHandler(stream_hander)

log_max_size = 10*1024*1024
log_file_count = 20

file_handler = handlers.RotatingFileHandler(filename = "test.log", maxBytes = log_max_size, backupCount=log_file_count)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# network 생성
logger.info("Network generation")
net = IENetwork(model = xml_path,weights = bin_path)
logger.info("Network generation Success")

# device (MYRIAD : NCS2)
logger.info("Device Init")
plugin = IEPlugin(device='MYRIAD')
logger.info("Device Init Success")

logger.info("Network Load...")
exec_net = plugin.load(net)
logger.info("Network Load Success")

def filtering_box(objects):
    # Filtering overlapping boxes
    # box를 걸러낸다
    objlen = len(objects)
    for i in range(objlen):
        # 신뢰도 == 0 skip
        if (objects[i].confidence == 0.0):
            continue
        
        for j in range(i + 1, objlen):
            # box가 많이 겹쳐져있다면 그중에 신뢰도가 높은 box를 뽑아 사용
            if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
                objects[j].confidence = 0.0
                
    return objects

def object_counting(frame,objects, obj_label = 0, drawing=False):  
    obj_count = 0
  
    # Drawing boxes
    # 박스 그리기
    for obj in objects:
        if obj.confidence > 0.2:
            label = obj.class_id
            if label == obj_label:
                obj_count += 1
          
            if drawing == True:
                label_text = LABELS[label] + " (" + "{:.1f}".format(obj.confidence * 100) + "%)"
                #frame boxing
                cv2.rectangle(frame, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), box_color, box_thickness)
                #frame label text
                cv2.putText(frame, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)

    return obj_count

def parsing(outputs):
    objects = []
    
    for output in outputs.values():
        objects = ParseYOLOV3Output(output, new_h, new_w, cam_h, cam_w, 0.5, objects, classes, coords, num)

    return filtering_box(objects)

def job():    
    start = time.time()
    frame = cv2.imread(img_path)
    
    resized_frame, prepimg = preprocess(frame,image_size,new_w,new_h)

    outputs = exec_net.infer({'inputs': prepimg}) # inference
    
    objects = parsing(outputs)
    
    count = object_counting(resized_frame, objects, obj_label = 0, drawing = drawing)
    
    logger.info("people count number : {}".format(count))
    
    end = time.time()
  
    logger.info("1 epoch time : {}".format(end - start))
    
    if drawing == True:
        cv2.imshow('image',resized_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def main():
    schedule.every(10).seconds.do(job)
    
    while True:
        schedule.run_pending()
    
    del net
    del exec_net
    del plugin


if __name__ == "__main__":
    main()



