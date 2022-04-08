import cv2
import time
import numpy as np
import multiprocessing
#import Jetson.GPIO as GPIO
#from jetcam.csi_camera import CSICamera

outputstate = True
output = False
calssid = 0

'''
output_pin1 = 37		#BOARD
output_pin2 = 35
output_pin3 = 33
output_pin4 = 31
def GPIO_init():
	global output_pin1
	global output_pin2
	global output_pin3
	global output_pin4
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(output_pin1, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(output_pin2, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(output_pin3, GPIO.OUT, initial=GPIO.LOW)
	GPIO.setup(output_pin4, GPIO.OUT, initial=GPIO.LOW)

def GPIO_OUTPIN(leab):
	global output_pin1
	global output_pin2
	global output_pin3
	global output_pin4
	high = GPIO.HIGH
	low = GPIO.LOW
	if leab == output_pin1:
		print("output ping board is: {}".format(output_pin1))
		GPIO.output(output_pin1, high)
		time.sleep(1)
		GPIO.output(output_pin1, low)
	elif leab == output_pin2:
		print("output ping board is: {}".format(output_pin2))
		GPIO.output(output_pin2, high)
		time.sleep(1)
		GPIO.output(output_pin2, low)
	elif leab == output_pin3:
		print("output ping board is: {}".format(output_pin3))
		GPIO.output(output_pin3, high)
		time.sleep(1)
		GPIO.output(output_pin3, low)
	elif leab == output_pin4:
		print("output ping board is: {}".format(output_pin4))
		GPIO.output(output_pin4, high)
		time.sleep(1)
		GPIO.output(output_pin4, low)
'''

def readimage():
	read = 1
	while(read):
		imag = cv2.imread('F:/CV/Project/SSD/ObjectDetectionOpenCV(MobileNetSSD)/test_mymodle/11.jpg',1)
		size = imag.shape[1] + imag.shape[2]
		if size>0 :
			read = 0
	return imag

def readclassname(path):    # 读取标签名
    classNames = []
    classFile = path
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    return classNames

def putconindence(imag, ind, con, boxs):    # 在原图像上写置信度数据
    for x in ind:
        boxx = boxs[x]
        print("confidence num {} confs is {}".format(x,round(con[x] * 100, 1)))
        cv2.putText(imag, str(round(con[x] * 100, 1)), (boxx[0]+5, boxx[1] + 40), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                (0, 0, 255), 1)

thres = 0.6 # Threshold to detect object
nms_thre = 0.2

#GPIO_init()
def main():
	img = readimage()
	cv2.imshow("Input",img)

	configPath = 'ssdmobilenetv3graph.pbtxt'
	weightsPath = 'tflite_graph.pb'
	calssnamePath = 'my.names'
	calssname = readclassname(calssnamePath)
	net = cv2.dnn_DetectionModel(weightsPath,configPath)
	net.setInputSize(320,320)
	net.setInputScale(1.0/ 127.5)
	net.setInputMean((127.5, 127.5, 127.5))
	net.setInputSwapRB(True)
	#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	classIds, confs, bbox = net.detect(img,confThreshold=thres)
	bbox = list(bbox)
	confs = list(np.array(confs).reshape(1,-1)[0])
	confs = list(map(float,confs))

	indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_thre)
	putconindence(img, indices, confs, bbox)
	for i in indices:
		num = i
		box = bbox[num]
		x,y,w,h = box[0],box[1],box[2],box[3]
		cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
		cv2.putText(img,calssname[classIds[i]-1].upper(),(box[0]+5,box[1]+20),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),1)
		print("image classid: ",classIds[num]-1)
	#GPIO_OUTPIN(output_pin4)
	#GPIO.cleanup()
	cv2.imshow("Output",img)
	while True:
		k = cv2.waitKey(10) & 0xFF
		if k == 27:
			break
	cv2.waitKey(1)
	cv2.destroyAllWindows()


if __name__=='__main__':
    main()