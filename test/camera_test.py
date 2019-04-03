import cv2
import numpy as np

cam_w = 320
cam_h = 240
image_size = 416

# cam에 맞는 size로 맞추기 위한 w,h
new_w = int(cam_w * min(image_size/cam_w, image_size/cam_h))
new_h = int(cam_h * min(image_size/cam_w, image_size/cam_h))

cam = cv2.VideoCapture(0)

while (True):
    ret, frame = cam.read()
    resized_image = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((image_size, image_size, 3), 128)
    canvas[(image_size - new_h) // 2:(image_size - new_h) // 2 + new_h,
    (image_size - new_w) // 2:(image_size - new_w) // 2 + new_w, :] = resized_image

    print(canvas)

    k = cv2.waitKey(10) & 0xFF

    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()