import cv2
import numpy as np
from PIL import Image
import os
import re  # re modülünü ekledik

path = "C://Users//Monster//Desktop//opencv_face_Identification_yuz_tanima_algoritmasi-master//yuz_tanima//veri//"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("C://Users//Monster//Desktop//opencv_face_Identification_yuz_tanima_algoritmasi-master//yuz_tanima//Cascade//haarcascade_frontalface_default.xml")

# Etiket çıkarma işlevini ekledik
def extract_label(file_name):
    match = re.search(r'\d+', file_name)  # Dosya adından sayıyı bulur
    if match:
        label = int(match.group())
        return label
    else:
        return None  # Eğer etiket bulunamazsa None döner

#Belirli bir dizinde bulunan yüz görüntülerini ve bu görüntülere karşılık gelen etiketleri (IDs) elde etmek için kullanıld
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        label = extract_label(os.path.split(imagePath)[-1])
        if label is not None:
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(label)
    return faceSamples, ids

print("\n Yüzler taranıyor. Birkaç saniye sürecek bekle...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('C://Users//Monster//Desktop//opencv_face_Identification_yuz_tanima_algoritmasi-master//yuz_tanima//trainer//trainer.yml')

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Veri Öğrenme Tamamlandı")