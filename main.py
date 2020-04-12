import numpy as np
import cv2
from playsound import playsound
import handleData
import os
import re
import random
import threading
from datetime import datetime 

cap = cv2.VideoCapture(2)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def removeFiles(path):
    for f in os.listdir(path):
        os.remove(f)

def random_with_N_digits(n):
    random.seed(datetime.now())
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return random.randint(range_start, range_end)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    faces = face_cascade.detectMultiScale(frame, 1.1, 10)
    for (x, y, w, h) in faces:

        roi = frame[y:y+h, x:x+w]
        fotopath = f"./faces/{random_with_N_digits(12)}.png"
        cv2.imwrite(fotopath, roi)
        if(handleData.train):
            x = threading.Thread(target=handleData.addTodataset, args=(fotopath, 3, 'trainpath'))
            x.start()
        else:
            name, percentage = handleData.testModel(fotopath)
            precstr = str(percentage) + "%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(frame, name, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(frame, precstr, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()