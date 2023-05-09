import os
import sys
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import detection
import ifaddr
from goprocam import GoProCamera, constants
import random
import time
from datetime import datetime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True).to(DEVICE)
model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True).to(DEVICE)
#model = detection.retinanet_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True).to(DEVICE)
model.eval()

class Hazard:
    def __init__(self, x, y, w, h):
        global frame
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xf = x+w
        self.yf = y+h
        self.kill = 0
        self.ident = str(datetime.now()) + str(x) + str(y) + str(w) + str(h)
        self.medianflowTracker = cv2.TrackerMedianFlow_create()
        #self.medianflowTracker = cv2.TrackerCSRT_create()
        self.bbox = (x, y, w, h)
        self.ok = self.medianflowTracker.init(frame, self.bbox)
        p1 = (int(self.bbox[0]), int(self.bbox[1]))
        p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
    def updatePosition(self):
        global frame
        self.ok, self.bbox = self.medianflowTracker.update(frame)
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
    global automobiles, pedestrians, frame
    if tipo == "car":
        for car in automobiles:
            p1 = (int(car.x), int(car.y))
            p2 = (int(car.x + car.w), int(car.y + car.h))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    elif tipo == "pedestrian":
        for pedestrian in pedestrians:
            p1 = (int(pedestrian.x), int(pedestrian.y))
            p2 = (int(pedestrian.x + pedestrian.w), int(pedestrian.y + pedestrian.h))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
    elif tipo == "all":
        drawHazards("car")
        drawHazards("pedestrian")

def sortDetections(detections, tipo):
    global automobiles, pedestrians
    if tipo == "car":
        for i in range(0, len(detections["boxes"])):
	        confidence = detections["scores"][i]
            #idx = int(detections["labels"][i])
	        if confidence > 0.8 and int(detections["labels"][i]) == 3:
		        box = detections["boxes"][i].detach().cpu().numpy(); (startX, startY, endX, endY) = box.astype("int"); automobiles.append(Automobile(startX, startY, abs(endX-startX), abs(endY-startY))); cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2);     
    elif tipo == "pedestrian":
        for i in range(0, len(detections["boxes"])):
	        confidence = detections["scores"][i]
            #idx = int(detections["labels"][i])
	        if confidence > 0.8 and int(detections["labels"][i]) == 1:
		        box = detections["boxes"][i].detach().cpu().numpy(); (startX, startY, endX, endY) = box.astype("int"); pedestrians.append(Pedestrian(startX, startY, abs(endX-startX), abs(endY-startY))); cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2);     
    elif tipo == "all":
        sortDetections(detections, "car")
        sortDetections(detections, "pedestrian")
        
#addr = ""
#for adapter in ifaddr.get_adapters():
#    if "GoPro" in adapter.nice_name:
#        for ip in adapter.ips:
#           if ip.is_IPv4:
#                addr = ip.ip.split(".")
#                addr[len(addr) - 1] = "51"
#                addr = ".".join(addr)

#gopro = GoProCamera.GoPro(ip_address=addr, camera=constants.gpcontrol)
#gopro.startWebcam(); gopro.video_settings(res='640p', fps='24'); gopro.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
#udp_addr = "udp://@{}:8554".format(addr)
#cap = cv2.VideoCapture(udp_addr, cv2.CAP_FFMPEG)
#while(cap.isOpened()== False):
#    print("Trying again...")
#    for adapter in ifaddr.get_adapters():
#        if "GoPro" in adapter.nice_name:
#            for ip in adapter.ips:
#                if ip.is_IPv4:
#                    addr = ip.ip.split(".")
#                    addr[len(addr) - 1] = "51"
#                    addr = ".".join(addr)

#    gopro = GoProCamera.GoPro(ip_address=addr, camera=constants.gpcontrol)
#    gopro.startWebcam(); gopro.video_settings(res='640p', fps='24'); gopro.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
#    udp_addr = "udp://@{}:8554".format(addr)
#    cap = cv2.VideoCapture(udp_addr, cv2.CAP_FFMPEG)

#print("Video Capture started")

cap = cv2.VideoCapture("elbanco.mp4")
ret, frame = cap.read()
print(frame.shape)
frame = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)
automobiles = []; pedestrians = []
iterations = 0

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)
        if iterations == 0:
            clearHazards("all")
            orig = frame.copy()
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame1 = frame1.transpose((2, 0, 1))
            frame1 = np.expand_dims(frame1, axis=0)
            frame1 = frame1 / 255.0
            frame1 = torch.FloatTensor(frame1)
            frame1 = frame1.to(DEVICE)
            detections = model(frame1)[0]
            frame = orig
            sortDetections(detections, "all")
            cv2.imshow('Frame',frame)
        else:
            updateHazards("all")
            clearFailedFilters("all")
            drawHazards("all")
            cv2.imshow('Frame',frame)
        iterations = iterations + 1
        if iterations == 30:
            iterations = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()