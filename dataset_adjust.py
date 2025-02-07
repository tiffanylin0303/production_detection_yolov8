import os
import shutil

train_img_dir = "dataset/train/images"
train_label_dir = "dataset/train/labels"

# record filenames 
img_file_list = []
for root, dirs, files in os.walk(train_img_dir, topdown = 0):
    for name in files:
        #print(name)
        img_file_list.append(name)

label_file_list = []
for root, dirs, files in os.walk(train_label_dir, topdown = 0):
    for name in files:
        label_file_list.append(name)

# make sure image filenames are equal to label filenames
for i in range(len(img_file_list)):
    if str(img_file_list[i][:-3]) != str(label_file_list[i][:-3]):
        print(i+1, img_file_list[i])
        print(i+1, label_file_list[i])

# move 172 files from train file to valid file
for i in range(172):
    shutil.move(os.path.join(train_img_dir,img_file_list[i]),os.path.join("dataset_test(1)/valid/images",img_file_list[i]))
    shutil.move(os.path.join(train_label_dir,label_file_list[i]), os.path.join("dataset_test(1)/valid/labels",label_file_list[i]))

# move 57 files from train file to test file
for i in range(57):
    shutil.move(os.path.join(train_img_dir,img_file_list[i+172]),os.path.join("dataset_test(1)/test/images",img_file_list[i+172]))
    shutil.move(os.path.join(train_label_dir,label_file_list[i+172]), os.path.join("dataset_test(1)/test/labels",label_file_list[i+172]))

print("finish")


