from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2 as cv
import time

xml_path = '/home/pi/workspace/IR/tiny-yolov3.xml'
bin_path = '/home/pi/workspace/IR/tiny-yolov3.bin'

# network 생성
net = IENetwork(model = xml_path,weights = bin_path)
'''
print("input : ",net.inputs)
print("input shape :",net.inputs['inputs'].shape)
print("output : ",net.outputs.keys())
print("output shape :",net.outputs['detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion'].shape)
print("output shape :",net.outputs['detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion'].shape)

print("net layer :",*list(net.layers.keys()),sep='\n')
'''
# device에 
plugin = IEPlugin(device='MYRIAD')
exec_net = plugin.load(net)

start = time.time()

frame = cv.imread('test.jpeg')
resized_image = cv.resize(frame, (416, 416), interpolation = cv.INTER_CUBIC)
prepimg = resized_image[np.newaxis, :, :, :]     # Batch size axis add
# position trans
prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW

end = time.time()

print('image process time : ',end - start)

start = time.time()

# inference
res = exec_net.infer({'inputs':prepimg})

end = time.time()

print('inference time : ',end-start)

print(res)
print(res.shape)
