import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C://Users//Monster//Desktop//opencv_face_Identification_yuz_tanima_algoritmasi-master//yuz_tanima//trainer//trainer.yml')
faceCascade = cv2.CascadeClassifier('C://Users//Monster//Desktop//opencv_face_Identification_yuz_tanima_algoritmasi-master//yuz_tanima//Cascade//haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None',"Bedo"]

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # id değerinin names listesinde bir index olarak geçerli olup olmadığını kontrol edin
        if id < len(names):
            id_name = names[id]
        else:
            id_name = "bilinmiyor"

        if confidence < 1000:
            id = id_name
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "bilinmiyor"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()