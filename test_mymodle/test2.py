import cv2
import numpy as np

thres = 0.5 # Threshold to detect object
nms_threshold = 0.2

def readimage():
    read = 1
    while ( read ):
        imag = cv2.imread('F:/CV/Project/SSD/ObjectDetectionOpenCV(MobileNetSSD)/test_mymodle/12.jpg', 1)
        size = imag.shape[1] + imag.shape[2]
        if size > 0:
            read = 0
    return imag


img = readimage()

classNames = []
classFile = 'my.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssdmobilenetv3graph3.pbtxt'
weightsPath = 'tflite_graph3.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=thres)
bbox = list(bbox)
confs = list(np.array(confs).reshape(1,-1)[0])
confs = list(map(float,confs))
# print(classIds,bbox)

indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

def putconindence(imag, ind, con, boxs):
    for x in ind:
        boxx = boxs[x]
        print("confidence num {} confs is {}".format(x,round(con[x] * 100, 1)))
        cv2.putText(imag, str(round(confs[x] * 100, 1)), (boxx[0]+5, boxx[1] + 40), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                    (0, 0, 255), 1)

putconindence(img, indices,confs,bbox)
for i in indices:
#	i = i[0]
	box = bbox[i]
	x,y,w,h = box[0],box[1],box[2],box[3]
	cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 0, 255), thickness=1)
	cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+5,box[1]+20),
								cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),1)
	print("image classid: ",classIds[i]-1)

cv2.imshow("Output", img)

while True:
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
cv2.waitKey(1)
cv2.destroyAllWindows()
