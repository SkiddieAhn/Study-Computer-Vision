import cv2
import numpy as np
import matplotlib.pyplot as plt

cap=cv2.VideoCapture('./data/vtest.avi')
if (not cap.isOpened()):
    print('Error opening video')

height,width=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

acc_gray=np.zeros(shape=(height,width),dtype=np.float32)
acc_bgr=np.zeros(shape=(height,width,3),dtype=np.float32)

t=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    t+=1
    print('t=1',t)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.accumulate(gray,acc_gray) # 누적 영상 구하기 
    avg_gray=acc_gray/t # t로 나누어서 평균영상 구하기 
    dst_gray=cv2.convertScaleAbs(avg_gray)

    cv2.accumulate(frame,acc_bgr)
    avg_bgr=acc_bgr/t
    dst_bgr=cv2.convertScaleAbs(avg_bgr)

    cv2.imshow('frame',frame)
    cv2.imshow('dst_gray',dst_gray)
    cv2.imshow('dst_bgr',dst_bgr)
    key=cv2.waitKey(20)
    if key == 27:
        break
    
if cap.isOpened():
    cap.release();


cv2.imwrite('./data/avg_gray.png',dst_gray)
cv2.imwrite('./data/avg_bgr.png',dst_bgr)
cv2.destroyAllWindows()

avg1=cv2.imread('./data/avg_gray.png',cv2.IMREAD_GRAYSCALE)
avg2=cv2.imread('./data/avg_bgr.png')
avg2rgb=cv2.cvtColor(avg2,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(avg1,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(avg2rgb)
