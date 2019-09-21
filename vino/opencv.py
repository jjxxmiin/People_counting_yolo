import cv2, time

# ===== OPENCV =====
xml_path = '/home/pi/workspace/IR/tiny-yolov3.xml'
bin_path = '/home/pi/workspace/IR/tiny-yolov3.bin'

cam = cv2.VideoCapture(0)

# Load the model
net = cv2.dnn.readNet(xml_path, bin_path)
# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

while (True):
    ret, frame = cam.read()
    cv2.imshow('image', frame)

    frame = cv2.resize(frame, (416, 416))

    blob = cv2.dnn.blobFromImage(frame, size=(416, 416), ddepth=cv2.CV_8U)
    net.setInput(blob)

    start = time.time()
    out = net.forward()
    end = time.time()

    print(end-start)
    print(out.shape)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
