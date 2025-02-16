import os
import DetectClass
import cv2

train_path = ['dataset/train/labels']
aug_path = ['dataset/train/labels']
image_path = ['dataset/train/images']
class_names = ['CPU_FAN_NO_Screws', 'CPU_FAN_Screw_loose', 'CPU_FAN_Screws', 'CPU_fan', 'CPU_fan_port', 'CPU_fan_port_detached', 'Incorrect_Screws', 'Loose_Screws', 'No_Screws', 'Scratch', 'Screws']
txt_content_list, img_class, img_list = DetectClass.data_augmentation_path(train_path)

for i in range(10):
    # read image
    image_path = img_list[i].replace("labels", "images").replace(".txt", ".jpg")
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    
    # read txt file
    label_path = img_list[i]
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    # draw bbox
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height
        
        # count coordinate
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # add the class tag
        label = class_names[class_id]
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # show image with bbox
    cv2.imshow('Image with Ground Truth', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # read aug image
    aug_image_path = f'dataset/aug/images/aug_{i}.jpg'
    aug_image = cv2.imread(aug_image_path)
    aug_image_height, aug_image_width = aug_image.shape[:2]
    
    # read aug txt file
    aug_label_path = f'dataset/aug/labels/aug.txt'
    with open(aug_label_path, 'r') as file:
        lines = file.readlines()
    
    # draw bbox
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height
        
        # count coordinate
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        cv2.rectangle(aug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
        # add the class tag
        aug_label = class_names[class_id]
        cv2.putText(aug_image, aug_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Image with Ground Truth', aug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    