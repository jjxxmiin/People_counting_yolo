from openvino.inference_engine import IENetwork, IEPlugin
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import cv2
import imutils
import math
import time
import os

# ============================hard coding===============================
cam_w = 320
cam_h = 240
image_size = 416
anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]

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

# cam에 맞는 size로 맞추기 위한 w,h
new_w = int(cam_w * min(image_size/cam_w, image_size/cam_h))
new_h = int(cam_h * min(image_size/cam_w, image_size/cam_h))

#xml_path = '/home/pi/workspace/IR/tiny-yolov3.xml'
#bin_path = '/home/pi/workspace/IR/tiny-yolov3.bin'

xml_path = "./IR/tiny-yolov3.xml" #<--- MYRIAD
bin_path = os.path.splitext(xml_path)[0] + ".bin"

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
    
      
# color setting
label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1
# =======================================================================

def preprocess(frame,image_size,new_x,new_y):
  resized_image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

  # 128로 채운다
  canvas = np.full((image_size, image_size, 3), 128)
  canvas[(image_size - new_h) // 2:(image_size - new_h) // 2 + new_h, (image_size - new_w) // 2:(image_size - new_w) // 2 + new_w, :] = resized_image

  prepimg = canvas

  prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
  prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW

  return prepimg

# detected object location
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

# EntryIndex
# location : output에서의 좌표
def EntryIndex(side, lcoords, lclasses, location, entry):
    # 몇번째 feature map
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

# inference output parsing *************
def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):
    out_blob_h = blob.shape[2] # 26 13
    out_blob_w = blob.shape[3] # 26 13

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
    # 26 * 26 / 13 * 13
    side_square = side * side

    # flatten : 일렬로 나열
    # 256 * 256
    # 256 * 169
    output_blob = blob.flatten()

    for i in range(side_square): 
        # 모든 좌표
        row = int(i / side) # i / 26 
        col = int(i % side) # i % 26
        for n in range(num):
            # object detection
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            # print(scale)
            # threshold보다 작으면 skip
            if (scale < threshold):
                continue
            # boxing x,y,w,h
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            ####################

            # classification
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * output_blob[class_index]
                # threshold보다 작으면 skip
                if prob < threshold:
                    continue
                # DetectionObject의 객체
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)
            #####################

    return objects

# box가 얼마나 겹쳐져있는지 확인 IOU
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


# network 생성
net = IENetwork(model = xml_path,weights = bin_path)

# device (MYRIAD : NCS2)
plugin = IEPlugin(device='MYRIAD')
exec_net = plugin.load(net)

# cam on/cam setting
cam = WebcamVideoStream(src=0).start()

while(True) :
  t1 = time.time()

  # frame preprossing
  frame = cam.read()
  
  frame = cv2.resize(frame, (cam_w, cam_h), interpolation=cv2.INTER_AREA)

  prepimg = preprocess(frame,image_size,new_w,new_h)

  start = time.time()
  # output : feature maps 확률
  outputs = exec_net.infer({'inputs': prepimg}) # inference
  end = time.time()

  # print('inference time : ', end - start)
  
  objects = []
  people_count = 0

  ########################
  # parsing
  # output : (1,256,13,13)
  # output : (1,256,26,26)
  ########################
  for output in outputs.values():
      # object : DetectionObject 타입
      objects = ParseYOLOV3Output(output, new_h, new_w, cam_h, cam_w, 0.4, objects)

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
              if objects[i].confidence < objects[j].confidence:
                  objects[i], objects[j] = objects[j], objects[i]
              objects[j].confidence = 0.0
  #########################


  # Drawing boxes
  # 박스 그리기
  for obj in objects:
      # 정확도가 20% 미만 skip
      if obj.confidence < 0.2:
          continue
      #라벨
      label = obj.class_id
      #사람만
      #if label == 0:
      #    people_count += 1
      #신뢰도
      confidence = obj.confidence
      label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
      #frame boxing
      cv2.rectangle(frame, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), box_color, box_thickness)
      #frame label text
      cv2.putText(frame, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)
  
  #frame fps text
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



