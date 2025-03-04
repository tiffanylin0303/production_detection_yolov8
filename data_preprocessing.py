import cv2
import DetectClass
train_path = ['dataset/train/labels']
aug_path = ['dataset/train/labels']
image_path = ['dataset/train/images']
class_names = ['CPU_FAN_NO_Screws', 'CPU_FAN_Screw_loose', 'CPU_FAN_Screws', 'CPU_fan', 'CPU_fan_port', 'CPU_fan_port_detached', 'Incorrect_Screws', 'Loose_Screws', 'No_Screws', 'Scratch', 'Screws']
txt_content_list, img_class, img_list, img_increase, img_increase_path = DetectClass.data_augmentation_path(train_path)

# Local Thresholding
# Read grayscale images
for i in range(10):
    image_path = img_list[i].replace("labels", "images").replace(".txt", ".jpg")
    original_image= cv2.imread(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Thresholding
    thresh_value = 128
    _, binary_image = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)

    # show original images and local thresholding images
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
