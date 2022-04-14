from sre_constants import SUCCESS
from time import time
import cv2
from matplotlib import image
import numpy as np
import time

np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath,modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.classesPath = classesPath
        self.modelPath = modelPath
        
        
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        
        self.readClasses()
        
    def readClasses(self):
        with open(self.classesPath, 'r') as f:
           self.classesList = f.read().splitlines()
            
           self.classesList.insert(0, "__background__")
            
           self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
       # print(self.classesList)    
        
    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return
        
        (success, image) = cap.read()
        
        startTime = 0
        
        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime
            
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.5)
            
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
                               
            bboxsIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold =0.5, nms_threshold = 0.2)
            
            if len(bboxsIdx) != 0:
                for i in range(0, len(bboxsIdx)):
                   
                   bbox = bboxs[np.squeeze(bboxsIdx[i])]
                   classConfidence = confidences[np.squeeze(bboxsIdx[i])]
                   classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxsIdx[i])])
                   classLabel = self.classesList[classLabelID]
                   classColor = [int(c) for c in self.colorList[classLabelID]]
                   
                   displayText = "{}: {:.2f}".format(classLabel, classConfidence)
                   
                   x,y,w,h = bbox
                   cv2.rectangle(image, (x,y), (x + w, y + h), color=classColor, thickness=1)
                   cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, 2)
                   
                   lineWidth = min(int(25 * 0.3), int(h * 0.3))
                   
                   cv2.line(image, (x + w, y), (x + w - lineWidth, y), classColor, thickness=4)
                   cv2.line(image, (x + w, y), (x + w, y + lineWidth), classColor, thickness=4)
                   
                   cv2.line(image, (x, y), (x + lineWidth, y), classColor, thickness=4)
                   cv2.line(image, (x, y), (x, y + lineWidth), classColor, thickness=4)
                   
                   cv2.line(image, (x + w, y + h), (x + w- lineWidth, y + h), classColor, thickness=4)
                   cv2.line(image, (x + w, y + h), (x + w, y + h - lineWidth), classColor, thickness=4)
                   
                   cv2.line(image, (x, y + h), (x + lineWidth, y + h), classColor, thickness=4)
                   cv2.line(image, (x, y + h), (x, y + h -lineWidth), classColor, thickness=4)
            
            cv2.putText(image, "FPS :" + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)       
            cv2.imshow("Result", image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            (success, image) = cap.read()
        cv2.destroyAllWindows()           
                               
                               
       
    