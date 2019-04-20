from threading import Thread
import cv2

class WebcamVideoStream:
	def __init__(self,src=0):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		
		self.stopped = False
	# Threading 시작
	def start(self):
		Thread(target=self.update,args=()).start()
		return self
	
	# 읽는부분
	def update(self):
		while True:
			if self.stopped:
				return
			(self.grabbed,self.frame) = self.stream.read()
			
	def read(self):
		return self.frame
		
	def stop(self):
		self.stopped = True
