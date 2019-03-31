import cv2 as cv
import time

xml_path = '/home/pi/workspace/IR/tiny-yolov3.xml'
bin_path = '/home/pi/workspace/IR/tiny-yolov3.bin'

# Load the model
net = cv.dnn.readNet(xml_path, bin_path)
# Specify target device 
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
   

# Read an image 
frame = cv.imread('test.jpeg')
frame = cv.resize(frame,(416,416))
      
# Prepare input blob and perform an inference 
blob = cv.dnn.blobFromImage(frame, size=(416, 416), ddepth=cv.CV_8U) 
net.setInput(blob) 
start = time.time()
out = net.forward()
end = time.time()

print("inference time : ",(end - start))


