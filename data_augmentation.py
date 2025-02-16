import os
import DetectClass
import matplotlib.pyplot as plt
from ultralytics import YOLO
import albumentations as A
import cv2
import os
import random
import shutil
import numpy as np

path = ['dataset/train/labels']
txt_content_list, img_class, img_list = DetectClass.data_augmentation_path(path)
print(img_class)
# YOLO 格式標籤的資料夾
AUG_IMAGES_DIR = "dataset/aug/images"
AUG_LABELS_DIR = "dataset/aug/labels"

# 確保增強後的目錄存在
os.makedirs(AUG_IMAGES_DIR, exist_ok=True)
os.makedirs(AUG_LABELS_DIR, exist_ok=True)

# 自訂增強策略 (Albumentations)
# augmentor1 = A.Compose([
#     A.HorizontalFlip(p=1),       # 水平翻轉
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# augmentor2 = A.Compose([
#     A.VerticalFlip(p=1),         # 垂直翻轉
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# augmentor3 = A.Compose([
#     A.HorizontalFlip(p=1),       # 水平翻轉
#     A.VerticalFlip(p=1),         # 垂直翻轉
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

augmentor4 = A.Compose([
    A.HorizontalFlip(p=0),       
    A.VerticalFlip(p=0),         
    # A.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.7, p=1),
    A.ToGray(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# augmentor5 = A.Compose([
#     A.GaussianBlur(blur_limit=3, p=0.5),
#     A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# 執行 Data Augmentation (710 train images)
for i in range(len(img_list)):
    img_path = img_list[i].replace("labels", "images").replace(".txt", ".jpg")
    label_path = img_list[i]

    # 讀取影像
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    # 讀取 YOLO 格式標籤
    with open(label_path, "r") as f:
        lines = f.readlines()

    bboxes = []
    category_ids = []
    
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        bboxes.append([x, y, w, h])
        category_ids.append(class_id)

    # 執行數據增強
    augmented = augmentor4(image=image, bboxes=bboxes, category_ids=category_ids)
    aug_img = augmented['image']
    aug_bboxes = augmented['bboxes']

    # 轉換標籤格式
    aug_label_txt = []
    for bbox, class_id in zip(aug_bboxes, category_ids):
        aug_label_txt.append(f"{class_id} " + " ".join(map(str, bbox)))

    # 儲存增強後的圖片
    aug_img_path = os.path.join(AUG_IMAGES_DIR, f'aug_{i}.jpg')
    aug_label_path = os.path.join(AUG_LABELS_DIR, f'aug_{i}.txt')

    cv2.imwrite(aug_img_path, aug_img)

    with open(aug_label_path, "w") as f:
        f.write("\n".join(aug_label_txt))
        
        
print("數據增強完成，已儲存到 augmented 資料夾")
