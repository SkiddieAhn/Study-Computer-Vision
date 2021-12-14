import cv2
"""
# Image Stitching
src1=cv2.imread('./data/stitch/stitch_image1.jpg')
src2=cv2.imread('./data/stitch/stitch_image2.jpg')
src3=cv2.imread('./data/stitch/stitch_image3.jpg')
src4=cv2.imread('./data/stitch/stitch_image4.jpg')
sticher=cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

status, dst2=sticher.stitch((src1,src2))
status, dst3=sticher.stitch((dst2,src3))
status, dst4=sticher.stitch((dst3,src4))

cv2.imshow('src1',src1)
cv2.imshow('dst2',dst2)
cv2.imshow('dst3',dst3)
cv2.imshow('dst4',dst4)
cv2.waitKey()
cv2.destroyAllWindows()"""

# Video Stitching
cap=cv2.VideoCapture('./data/stitch/stitch_videoInput.mp4')
t=0
images=[]
STEP=20

while True:
    t+=1
    retval, frame=cap.read()
    if not retval:
        break
    img=cv2.resize(frame,dsize=(640,480))
    #img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    # 20프레임 당 하나씩 가져와서 어펜딩 
    if t%STEP ==0:
        images.append(img)

    cv2.imshow('img',img)
    key=cv2.waitKey(25)
    if key == 27: #Esc
        break

# 스티칭 작업 
print('len(images)=',len(images))
stitcher=cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
status,dst=stitcher.stitch(images)
if status== cv2.STITCHER_OK:
    cv2.imshow('dst1',dst)
    cv2.waitKey()

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()

