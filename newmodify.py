 #Based on Zed code - Person Fall detection using raspberry pi camera and opencv lib. Link.
https://web-chat.global.assistant.watson.cloud.ibm.com/preview.html?region=eu-gb&integrationID=95e96f34-5bc2-4345-bc72-9d51816b81f5&serviceInstanceID=36699238-7143-45f7-9106-391294718c42
import cv2
import time

cap = cv2.VideoCapture("queda.mp4")
time.sleep(2)

fgbg = cv2.createBackgroundSubtractorMOG2()
j = 0
while(1):
    ret, frame = cap.read()
    
    #Conver each frame to gray scale and subtract the background
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    
    #Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        areas = []

        for contour in contours:
            ar = cv2.contourArea(contour)
            areas.append(ar)
        
        max_area = max(areas or [0])

        max_area_index = areas.index(max_area)

        cnt = contours[max_area_index]

        M = cv2.moments(cnt)
        
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0)
        
        if h < w:
            j += 1
            
        if j > 10:
            #print ("FALL")
            #cv2.putText(fgmask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        if h > w:
            j = 0 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


        cv2.imshow('video', frame)
    
        if cv2.waitKey(1)&0xff==ord('q'):
            break
    else:
            break
        
        
cap.release()
cv2.destroyAllWindows()
