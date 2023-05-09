import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration
from start_cameras import Start_Cameras
import os
import sys
import random
import matplotlib.pyplot as plt
import time
from datetime import datetime


# Depth map default preset
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 160
TTH = 100
UR = 10
SR = 14
SPWS = 100


def load_map_settings(file):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings, sbm
    print('Loading parameters from file...')
    f = open(file, 'r')
    data = json.load(f)
    #loading data from the json file and assigning it to the Variables
    SWS = data['SADWindowSize']
    PFS = data['preFilterSize']
    PFC = data['preFilterCap']
    MDS = data['minDisparity']
    NOD = data['numberOfDisparities']
    TTH = data['textureThreshold']
    UR = data['uniquenessRatio']
    SR = data['speckleRange']
    SPWS = data['speckleWindowSize']
    
    #changing the actual values of the variables
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=SWS) 
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print('Parameters loaded from file ' + file)

def stereo_depth_map(rectified_pair):
    #blockSize is the SAD Window Size

    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    image = np.array(disparity_normalized, dtype = np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return disparity_color, disparity_normalized

def onMouse(event, x, y, flag, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disparity_normalized[y][x]
        print("Distance in centimeters {}".format(distance))
        return distance
        
class Hazard:
    def __init__(self, x, y, w, h):
        global finalLeftFrame
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xf = x+w
        self.yf = y+h
        self.kill = 0
        self.ident = str(datetime.now()) + str(x) + str(y) + str(w) + str(h)
        self.medianflowTracker = cv2.TrackerMedianFlow_create()
        self.bbox = (x, y, w, h)
        self.ok = self.medianflowTracker.init(finalLeftFrame, self.bbox)
        p1 = (int(self.bbox[0]), int(self.bbox[1]))
        p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
    def updatePosition(self):
        global finalLeftFrame
        self.ok, self.bbox = self.medianflowTracker.update(finalLeftFrame)
        if self.ok:
            self.x = self.bbox[0]
            self.y = self.bbox[1]
            self.w = self.bbox[2]
            self.h = self.bbox[3]
            self.xf = self.x+self.w
            self.yf = self.y+self.h
        else:
            self.kill = 1
    def getID(self):
        return self.ident
    def __del__(self):
        return

class Automobile(Hazard):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)

class Pedestrian(Hazard):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        

def clearHazards(tipo):
    global automobiles, pedestrians
    if tipo == "car": 
        for car in automobiles:
            del car
        automobiles = []
    elif tipo == "pedestrian":
        for pedestrian in pedestrians:
            del pedestrian
        pedestrians = []
    elif tipo == "all":
        clearHazards("car")
        clearHazards("pedestrian")

def clearFailedFilters(tipo):
    global automobiles, pedestrians
    if tipo == "car":
        for i in range(len(automobiles)):
            if i < len(automobiles):
                if automobiles[i].kill == 1:
                    del automobiles[i]
                    i = i - 1
    elif tipo == "pedestrian":
        for i in range(len(pedestrians)):
            if i < len(pedestrians):
                if pedestrians[i].kill == 1:
                    del pedestrians[i]
                    i = i - 1
    elif tipo == "all":
        clearFailedFilters("car")
        clearFailedFilters("pedestrian")

def updateHazards(tipo):
    global automobiles, pedestrians
    if tipo == "car":
        for car in automobiles:
            car.updatePosition()
    elif tipo == "pedestrian":
        for pedestrian in pedestrians:
            pedestrian.updatePosition()
    elif tipo == "all":
        updateHazards("car")
        updateHazards("pedestrian")

def drawHazards(tipo):
    global automobiles, pedestrians, finalLeftFrame, finalDepthMap
    if tipo == "car":
        for car in automobiles:
            p1 = (int(car.x), int(car.y))
            p2 = (int(car.x + car.w), int(car.y + car.h))
            cv2.rectangle(finalLeftFrame, p1, p2, (255,0,0), 2, 1)
            cv2.rectangle(finalDepthMap, p1, p2, (255,0,0), 2, 1)
    elif tipo == "pedestrian":
        for pedestrian in pedestrians:
            p1 = (int(pedestrian.x), int(pedestrian.y))
            p2 = (int(pedestrian.x + pedestrian.w), int(pedestrian.y + pedestrian.h))
            cv2.rectangle(finalLeftFrame, p1, p2, (0,255,0), 2, 1)
            cv2.rectangle(finalDepthMap, p1, p2, (0,255,0), 2, 1)
    elif tipo == "all":
        drawHazards("car")
        drawHazards("pedestrian")

def sortDetections(detections):
    global automobiles, pedestrians, finalLeftFrame, finalDepthMap
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        classification = detections[0,0,i,1]
        idx = int(detections[0,0,i,1])
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype('int')
        if classification == 7 and confidence > 0.49:
            automobiles.append(Automobile(startX, startY, abs(endX-startX), abs(endY-startY)))
            cv2.rectangle(finalLeftFrame, (startX, startY), (endX, endY), (255,0,0),3)
            cv2.rectangle(finalDepthMap, (startX, startY), (endX, endY), (255,0,0),3)
        if classification == 15 and confidence > 0.49:
            pedestrians.append(Pedestrian(startX, startY, abs(endX-startX), abs(endY-startY)))
            cv2.rectangle(finalLeftFrame, (startX, startY), (endX, endY), (0,255,0),3)
            cv2.rectangle(finalDepthMap, (startX, startY), (endX, endY), (0,255,0),3)
h, w = 360, 640
automobiles = []; pedestrians = []
iterations = 0
net = cv2.dnn.readNetFromCaffe("Chuanqi/deploy.prototxt","Chuanqi/mobilenet_iter_73000.caffemodel")

left_camera = Start_Cameras(0).start()
right_camera = Start_Cameras(1).start()
load_map_settings("../3dmap_set.txt")

#cv2.namedWindow("DepthMap")

left_grabbed, left_frame = left_camera.read()
right_grabbed, right_frame = right_camera.read()
finalLeftFrame = cv2.flip(cv2.flip(right_frame, 0), 1)
finalRightFrame = cv2.flip(cv2.flip(left_frame, 0), 1)
finalDepthMap = 0
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (2*w,h))
while True:
    left_grabbed, left_frame = left_camera.read()
    right_grabbed, right_frame = right_camera.read()

    if left_grabbed and right_grabbed:  
        #Convert BGR to Grayscale     
        left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        #calling all calibration results
        calibration = StereoCalibration(input_folder='../calib_result')
        rectified_pair = calibration.rectify((left_gray_frame, right_gray_frame))
        disparity_color, disparity_normalized = stereo_depth_map(rectified_pair)

        #Mouse clicked function
        cv2.setMouseCallback("DepthMap", onMouse, disparity_normalized)

        #Show depth map and image frames
        output = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
        finalDepthMap = cv2.flip(cv2.flip(disparity_color, 0), 1)
        finalOutput = cv2.flip(cv2.flip(output, 0), 1)
        finalLeftFrame = cv2.flip(cv2.flip(right_frame, 0), 1)
        finalRightFrame = cv2.flip(cv2.flip(left_frame, 0), 1)
        frameResize = cv2.resize(finalLeftFrame, (300, 300))
        if iterations == 0:
            clearHazards("all")
            blob = cv2.dnn.blobFromImage(frameResize, size=(300, 300), ddepth=cv2.CV_8U)
            net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5,127.5,127.5])
            detections=net.forward()
            sortDetections(detections)
        else:
            updateHazards("all")
            clearFailedFilters("all")
            drawHazards("all")
        #cv2.imshow("DepthMap", finalDepthMap)
        #cv2.imshow("Frames", np.hstack((finalLeftFrame, finalRightFrame, finalDepthMap)))
        cv2.imshow("Frames", np.hstack((finalLeftFrame, finalDepthMap)))
        out.write(np.hstack((finalLeftFrame, finalDepthMap)))
        if iterations == 2:
            iterations = -1
        iterations = iterations + 1
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        else:
            continue

left_camera.stop()
left_camera.release()
right_camera.stop()
right_camera.release()
out.release()
cv2.destroyAllWindows()



    


