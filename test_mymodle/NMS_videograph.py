import cv2
import numpy as np
import time
#import multiprocessing

thres = 0.7             # 置信度阈值
nms_threshold = 0.2     # NMS阈值
outputstate = True
output = False
calssid = 0

def waitcapture(capture):   # 等待摄像头开启
    read = 1
    while ( read ):
        ret, imag = capture.read()
        size = imag.shape[1] + imag.shape[2]
        if size > 0:
            read = 0

def putconindence(imag, ind, con, boxs):    # 在原图像上写置信度数据
    for x in ind:
        boxx = boxs[x]
        print("confidence num {} confs is {}".format(x,round(con[x] * 100, 1)))
        cv2.putText(imag, str(round(con[x] * 100, 1)), (boxx[0]+5, boxx[1] + 40), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                (0, 0, 255), 1)

def readclassname(path):    # 读取标签名
    classNames = []
    classFile = path
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    return classNames

def classidoutput(id):
    global output
    global calssid
    global outputstate
    while outputstate :
        if output == True :
            time.sleep(0.5)
            print("image classid: ",calssid)
            time.sleep(1)
            print("IO output is down! ")
            output = False


def main() :
    global outputstate
    global calssname
    global output
    cap = cv2.VideoCapture(0)
    configPath = 'ssdmobilenetv3graph3.pbtxt'
    weightsPath = 'tflite_graph3.pb'
    calssnamePath = 'my.names'
    calssname = readclassname(calssnamePath)

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    waitcapture(cap)
    while True:
        ret, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))
        # print(classIds,bbox)

        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
        putconindence(img, indices,confs,bbox)
        for i in indices:
            #	i = i[0]
	        box = bbox[i]
	        x,y,w,h = box[0],box[1],box[2],box[3]
	        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 0, 255), thickness=1)
	        cv2.putText(img,calssname[classIds[i]-1].upper(),(box[0]+5,box[1]+20),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),1)
            #output = True
        cv2.imshow("Output", img)
        k = cv2.waitKey(10) & 0xFF
        if k == 27 :
            break
    cv2.destroyAllWindows()
    #outputstate = False

if __name__=='__main__':
    main()
#    out  = multiprocessing.Process(target= classidoutput, name= "output")
#    out.start()
#    out.join( )
