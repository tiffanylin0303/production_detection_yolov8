import os

path = ['dataset/train/labels']
image_path = ['dataset/train/images']

# check whether the sublist contains "10", "2", "0" (frequently appearing categories in dataset)
def contains_forbidden_strings(s):
    return any(x in s for x in ["10", "2"])

# check whether the sublist contains "7", "5", "6", "9", "1" (seldom appearing categories in dataset)
def contains_need_strings(s):
    return any(x in s for x in ["7", "5", "6", "9", "1"])

def data_augmentation_path(path):
    # record filenames 
    txt_content_list = []
    img_list = []
    content = []
    img_path = ""
    # get dataset annotation list 
    for i in range(len(path)):
        for root, dirs, files in os.walk(path[i], topdown = 0):
            for name in files:
                img_path = str(path[i]) + "/" + str(name)
                img_list.append(img_path)
                f = open((img_path), 'r')
                for line in f.readlines():
                    if line[1] != " ":
                        n = line[:2]
                    else:
                        n = line[0]
                    content.append(n)
                txt_content_list.append(content)
                content = []
                f.close()
    
    img_class = [0]*11  # count raw dataset annotation class distribution
    img_increase = []   # do data augmentation image class list
    img_increase_path = []  # do data augmentation label path
    img_increase_list = [] # do data augmentation image path
    for i in range(len(txt_content_list)):
        for j in range(len(txt_content_list[i])):
            k = int(txt_content_list[i][j])
            img_class[k] += 1
            
        # make sure the images which need to do data augmentation contains annotations seldom appearing and don't have often appearing classes
        if all(not contains_forbidden_strings(x) for x in txt_content_list[i]) and \
            any(contains_need_strings(y) for y in txt_content_list[i]):
            img_increase.append(txt_content_list[i])
            img_increase_path.append(img_list[i])   #txt file append
            new_path = img_list[i].replace("labels", "images").replace(".txt", ".jpg")
            if os.path.exists(new_path):
                img_increase_list.append(new_path)
            else:
                print(f"File does not exist: {new_path}")

    return txt_content_list, img_class, img_list, img_increase, img_increase_path

if __name__ == "__main__":
    data_augmentation_path(['dataset/train/labels'])
