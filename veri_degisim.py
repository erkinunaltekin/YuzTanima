import cv2
import os

# Klasör yolu
folder_path = r'C:\Users\Monster\Desktop\opencv_face_Identification_yuz_tanima_algoritmasi-master\yuz_tanima\veri'

# Yeni boyut
new_width = 128
new_height = 128

# Klasördeki tüm resimleri listele
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):  # Sadece jpg uzantılı dosyaları işle
        file_path = os.path.join(folder_path, filename)
        
        # Görüntüyü yükle
        image = cv2.imread(file_path)
        
        if image is not None:
            # Görüntüyü aynalama (horizontal flip)
            flipped_image = cv2.flip(image, 1)
            
            # Görüntüyü yeniden boyutlandır
            resized_image = cv2.resize(image, (new_width, new_height))
            
            # Görüntüyü döndürme
            rotation_angle = 45  # Derece cinsinden döndürme açısı
            rows, cols, _ = image.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
            
            # Parlaklık artırma
            brightness_factor = 1.5  # Parlaklık faktörü
            brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
            
            # Kontrast artırma
            contrast_factor = 1.5  # Kontrast faktörü
            contrasted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
            
            # Yeni görüntüleri kaydet
            base_filename = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(folder_path, f'{base_filename}_flipped.jpg'), flipped_image)
            cv2.imwrite(os.path.join(folder_path, f'{base_filename}_resized.jpg'), resized_image)
            cv2.imwrite(os.path.join(folder_path, f'{base_filename}_rotated.jpg'), rotated_image)
            cv2.imwrite(os.path.join(folder_path, f'{base_filename}_brightened.jpg'), brightened_image)
            cv2.imwrite(os.path.join(folder_path, f'{base_filename}_contrasted.jpg'), contrasted_image)
print("Veri Değiştirme Tamamlandı")