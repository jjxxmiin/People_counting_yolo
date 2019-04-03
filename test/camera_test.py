import cv2
import numpy as np
import imutils

def set_res(cam, x,y):
  cam.set(3, x)
  cam.set(4, y)

cam_w = 1024
cam_h = 720
image_size = 416

# cam에 맞는 size로 맞추기 위한 w,h
new_w = int(cam_w * min(image_size/cam_w, image_size/cam_h))
new_h = int(cam_h * min(image_size/cam_w, image_size/cam_h))

cam = cv2.VideoCapture(0)
set_res(cam,1280,720)

while (True):
    ret, frame = cam.read()
    
    resized_image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    cv2.imshow('window',frame)

    k = cv2.waitKey(10) & 0xFF

    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
