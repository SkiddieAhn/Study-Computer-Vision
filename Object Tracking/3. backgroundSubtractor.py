import cv2
import numpy as np
import matplotlib.pyplot as plt

cap=cv2.VideoCapture('./data/vtest.avi')
if (not cap.isOpened()):
    print('Error opening video')

height,width=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

bgMog1=cv2.createBackgroundSubtractorMOG2()
bgMog2=cv2.createBackgroundSubtractorMOG2(varThreshold=25,detectShadows=False) # 임계치값 낮추면 foreground 많이 생성 
bgKnn1=cv2.createBackgroundSubtractorKNN()
bgKnn2=cv2.createBackgroundSubtractorKNN(dist2Threshold=1000,detectShadows=False) # 임계치값 낮추면 foreground 많이 생성

AREA_TH=80
def findObjectAndDraw(bImage,src):
    res=src.copy()
    bImage=cv2.erode(bImage,None,5)
    bImage=cv2.dilate(bImage,None,5)
    bImage=cv2.erode(bImage,None,7)
    contours,_=cv2.findContours(bImage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src,contours,-1,(255,0,0,),1)
    for i,cnt in enumerate(contours):
        area=cv2.contourArea(cnt) 
        if area>AREA_TH: 
            x,y,width,height=cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+width,y+height),(0,0,255),2)
        return res

t=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    t+=1
    print('t=',t)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(frame,(5,5),0,0)

    bImage1=bgMog1.apply(blur)
    bImage2=bgMog2.apply(blur)
    bImage3=bgKnn1.apply(blur)
    bImage4=bgKnn2.apply(blur)
    dst1=findObjectAndDraw(bImage1,frame)
    dst2=findObjectAndDraw(bImage2,frame)
    dst3=findObjectAndDraw(bImage3,frame)
    dst4=findObjectAndDraw(bImage4,frame)

    cv2.imshow('bImage1',bImage1)
    cv2.imshow('bgMog1',dst1)
    cv2.imshow('bImage2',bImage2)
    cv2.imshow('bgMog2',dst2)
    cv2.imshow('bImage3',bImage3)
    cv2.imshow('bgKnn1',dst3)
    cv2.imshow('bImage4',bImage4)
    cv2.imshow('bIKnn2',dst4)
    key=cv2.waitKey(25)
    if key==27:
        break
    
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
