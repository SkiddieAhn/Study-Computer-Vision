import cv2
import numpy as np
import matplotlib.pyplot as plt

cap=cv2.VideoCapture('./data/vtest.avi')
if (not cap.isOpened()):
    print('Error opening video')

height,width=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))


TH=40
AREA_TH=80
bkg_gray=cv2.imread('./data/avg_gray.png',cv2.IMREAD_GRAYSCALE)
bkg_bgr=cv2.imread('./data/avg_bgr.png')

mode=cv2.RETR_EXTERNAL
method=cv2.CHAIN_APPROX_SIMPLE

t=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    t+=1
    print('t=',t)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    diff_gray=cv2.absdiff(gray,bkg_gray) # |gray(현재 프레임)-bkg_gray(고정된 배경)|저장 -> object만 남음 
    diff_bgr=cv2.absdiff(frame,bkg_bgr)
    db,dg,dr=cv2.split(diff_bgr) # 채널별로 나뉨 
    fet,bb=cv2.threshold(db,TH,255,cv2.THRESH_BINARY) # threshold를 줌 -> 두 영상의 차이가 40이상인 경우만 남
    ret,bg=cv2.threshold(dg,TH,255,cv2.THRESH_BINARY)
    ret,br=cv2.threshold(dr,TH,255,cv2.THRESH_BINARY)

    bImage=cv2.bitwise_or(bb,bg) # (b채널, g채널, r채널)에서 각각 밝기값의 차이가 큰 애들만 남겨서 합침 (bImage)
    bImage=cv2.bitwise_or(br,bImage)

    # 모폴로지 연산 
    bImage=cv2.erode(bImage,None,5) # 노이즈 제거
    bImage=cv2.dilate(bImage,None,5) # 구멍 없애기
    bImage=cv2.erode(bImage,None,7) # 노이즈 제거


    contours,hierarchy=cv2.findContours(bImage,mode,method) # 외곽선 구하기 
    cv2.drawContours(frame,contours,-1,(255,0,0),1) # 프레임에 외곽선 그리기 
    for i,cnt in enumerate(contours):
        area=cv2.contourArea(cnt) # 컨투어별로 컨투어 area를 구함 (오브젝트를 감싸는 area)
        if area>AREA_TH: # area 80보다 큰 경우만 바운딩 박스 구함 
            x,y,width,height=cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+width,y+height),(0,0,255),2)

        cv2.imshow('frame',frame)
        cv2.imshow('bImage',bImage)
        cv2.imshow('diff_gray',diff_gray)
        cv2.imshow('diff_bgr',diff_bgr)
        key=cv2.waitKey(25)
        if key == 27:
            break

if cap.isOpened():
    cap.release();
cv2.destoryAllWindows()
