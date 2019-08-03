from openvino.inference_engine import IENetwork, IEPlugin
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import cv2
import imutils
import math
import time
# multiprocessing을 위한 유용한 모듈
import multiprocessing as mp
from tqdm import tqdm
# 실시간 추적 모듈 sort 사용
from sort import *

tracker = Sort()
memory = {}
# ============================hard coding===============================
cam_w = 640
cam_h = 360
image_size = 416
anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]
#anchors = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]

fps = ""
framepos = 0
frame_count = 0
vidfps = 0
skip_frame = 0
elapsedTime = 0

yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

classes = 80
coords = 4
num = 3

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
    
    
      
label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1
# =======================================================================

def preprocess(frame,image_size,new_x,new_y):
  resized_image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

  canvas = np.full((image_size, image_size, 3), 128)
  canvas[(image_size - new_h) // 2:(image_size - new_h) // 2 + new_h, (image_size - new_w) // 2:(image_size - new_w) // 2 + new_w, :] = resized_image

  prepimg = canvas

  prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
  prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW

  return prepimg

class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence

def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):
    out_blob_h = blob.shape[2] 
    out_blob_w = blob.shape[3] 

    side = out_blob_h
    anchor_offset = 0

    if len(anchors) == 18:   ## YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    elif len(anchors) == 12: ## tiny-YoloV3 *********
        if side == yolo_scale_13:
            anchor_offset = 2 * 3
        elif side == yolo_scale_26:
            anchor_offset = 2 * 0

    else:                    ## ???
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0
    side_square = side * side

    output_blob = blob.flatten()

    for i in range(side_square): 
        row = int(i / side) 
        col = int(i % side)
        for n in range(num):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if (scale < threshold):
                continue
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            
            
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * output_blob[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)
            

    return objects

def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval


net = IENetwork(model = xml_path,weights = bin_path)

plugin = IEPlugin(device='MYRIAD')
exec_net = plugin.load(net)

cam = WebcamVideoStream(src=0).start()

while(True) :
  t1 = time.time()

  frame = cam.read()
  
  frame = cv2.resize(frame, (cam_w, cam_h), interpolation=cv2.INTER_AREA)

  prepimg = preprocess(frame,image_size,new_w,new_h)

  start = time.time()
  outputs = exec_net.infer({'inputs': prepimg}) # inference
  end = time.time()
  
  objects = []

  for output in outputs.values():
      objects = ParseYOLOV3Output(output, new_h, new_w, cam_h, cam_w, 0.4, objects)

  objlen = len(objects)
  for i in range(objlen):
      if (objects[i].confidence == 0.0):
          continue
      
      for j in range(i + 1, objlen):
          if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
              if objects[i].confidence < objects[j].confidence:
                  objects[i], objects[j] = objects[j], objects[i]
              objects[j].confidence = 0.0
              
  dets = []
  boxes = []
  people_count = 0
  
  for i in range(0,len(objects)):
    if objects[i].confidence < 0.2:
            continue
    dets.append([objects[i].xmin, objects[i].ymin , objects[i].xmax, objects[i].ymax , objects[i].confidence])

  dets = np.asarray(dets)
  
  if dets.size > 0:
    tracks = tracker.update(dets)
      
    for trk in tracks:
        cv2.rectangle(frame, (int(trk[0]), int(trk[1])), (int(trk[2]), int(trk[3])), box_color, box_thickness)
  
  
  cv2.putText(frame, fps, (cam_w - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
  cv2.imshow("Result", frame)

  if cv2.waitKey(1)&0xFF == ord('q'):
      break
  elapsedTime = time.time() - t1
  fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)


cv2.destroyAllWindows()
cam.stop()
del net
del exec_net
del plugin



