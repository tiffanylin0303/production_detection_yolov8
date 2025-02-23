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
txt_content_list, img_class, img_list, img_increase, img_increase_path = DetectClass.data_augmentation_path(path)
print(img_class)

AUG_IMAGES_DIR = "dataset/aug/images"
AUG_LABELS_DIR = "dataset/aug/labels"

# make sure the path of data augmentation exist
os.makedirs(AUG_IMAGES_DIR, exist_ok=True)
os.makedirs(AUG_LABELS_DIR, exist_ok=True)

# data augmentation ways (Albumentations)
# augmentor1 = A.Compose([
#     A.HorizontalFlip(p=1),       # flip horizontally
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# augmentor2 = A.Compose([
#     A.VerticalFlip(p=1),         # flip vertically
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# augmentor3 = A.Compose([
#     A.HorizontalFlip(p=1),       
#     A.VerticalFlip(p=1),         
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# augmentor4 = A.Compose([
#     A.HorizontalFlip(p=0),       
#     A.VerticalFlip(p=0),         
#     A.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.7, p=0.25),
#     A.ToGray(p=0.5),
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

augmentor5 = A.Compose([
    A.HorizontalFlip(p=0),       
    A.VerticalFlip(p=0), 
    A.GaussianBlur(blur_limit=3, p=1),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

augmentor6 = A.Compose([
    A.HorizontalFlip(p=0),       
    A.VerticalFlip(p=0), 
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# Exrcute data augmentation
for i in range(len(img_increase_path)):
    img_path = img_increase_path[i].replace("labels", "images").replace(".txt", ".jpg")
    label_path = img_increase_path[i]

    # read images
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    # read labels
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

    # transform into yolo label
    aug_label_txt = []
    for bbox, class_id in zip(bboxes, category_ids):
        aug_label_txt.append(f"{class_id} " + " ".join(map(str, bbox)))

    # photo shapen way1 -- unsharp masking (USM) with laplacian (拉普拉斯)
    image = cv2.imread(img_path)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    sharpened_img = cv2.addWeighted(image, 1.2, laplacian, -0.2, 0)

    # save data augmentation images
    aug_img_path = img_path.replace(".jpg", "_sharpen(laplacian).jpg")
    aug_label_path = label_path.replace(".txt", "_sharpen(laplacian).txt")

    cv2.imwrite(aug_img_path, sharpened_img)

    with open(aug_label_path, "w") as f:
        f.write("\n".join(aug_label_txt))
    
    # # photo shapen way2 -- unsharp masking (USM) with GaussianBlur (高斯模糊)
    # # GaussianBlur
    # blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # # USM
    # sharpened_img = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    
    # # cv2.imshow('Original Image', image)
    # # cv2.imshow('USM Enhanced Image', sharpened_img)
    
    # # save data augmentation images
    # aug_img_path = img_path.replace(".jpg", "_sharpen(Gauss).jpg")
    # aug_label_path = label_path.replace(".txt", "_sharpen(Gauss).txt")

    # cv2.imwrite(aug_img_path, sharpened_img)

    # with open(aug_label_path, "w") as f:
    #     f.write("\n".join(aug_label_txt))
print("數據增強完成")
