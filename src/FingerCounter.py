import cv2
import os
import time

from HandTrackingModule import handDetector


#########################
wCam, hCam = 640, 480
##########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'fingers'
myList = os.listdir(folderPath)
overlayList = []
for imgPath in sorted(myList):
    image = cv2.imread(f'{folderPath}/{imgPath}')
    resized_image = cv2.resize(image, (200, 200), 
                               interpolation=cv2.INTER_AREA)
    overlayList.append(resized_image)

pTime = 0

detector = handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findhands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 Fingers
        for id_ in range(1,5):
            if lmList[tipIds[id_]][2] < lmList[tipIds[id_]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)

        totalFingers = fingers.count(1)
        img[0:200, 0:200] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), 
                cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (500, 30), 
            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(1)
