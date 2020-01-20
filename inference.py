import cv2
import math
import numpy as np

def preprocess(frame,image_size,new_w,new_h):
	resized_image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

	# 128로 채운다
	canvas = np.full((image_size, image_size, 3), 128)
	canvas[(image_size - new_h) // 2:(image_size - new_h) // 2 + new_h, (image_size - new_w) // 2:(image_size - new_w) // 2 + new_w, :] = resized_image

	prepimg = canvas

	prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
	prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW

	return resized_image, prepimg

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
def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects, classes, coords,num):
    anchors = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
    
    out_blob_h = blob.shape[2] # 26 13
    out_blob_w = blob.shape[3] # 26 13

    side = out_blob_h
    anchor_offset = 0

    if side == 13:
        anchor_offset = 2 * 6
    elif side == 26:
        anchor_offset = 2 * 3
    elif side == 52:
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
