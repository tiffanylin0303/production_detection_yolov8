import os

path = ['dataset_test/train/labels', 'dataset_test/valid/labels', 'dataset_test/test/labels']

# check whether the sublist contains "10", "2", "0" (frequently appearing categories in dataset)
def contains_forbidden_strings(s):
    return any(x in s for x in ["10", "2"])

# check whether the sublist contains "7", "5", "6", "9", "1" (seldom appearing categories in dataset)
def contains_need_strings(s):
    return any(x in s for x in ["7", "5", "6", "9", "1"])

def data_agumentation_path(path):
    # record filenames 
    txt_content_list = []
    img_list = []
    content = []
    
    # get dataset annotation list 
    for i in range(len(path)):
        for root, dirs, files in os.walk(path[i], topdown = 0):
            for name in files:
                img_list.append(str(os.path.join(path[i],name)))
                f = open(os.path.join(path[i],name), 'r')
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
    img_increase_list = [] # do data augmentation image path
    for i in range(len(txt_content_list)):
        for j in range(len(txt_content_list[i])):
            k = int(txt_content_list[i][j])
            img_class[k] += 1
            
        # make sure the images which need to do data augmentation contains annotations seldom appearing and don't have often appearing classes
        if all(not contains_forbidden_strings(x) for x in txt_content_list[i]) and \
            all(contains_need_strings(y) for y in txt_content_list[i]):
            img_increase.append(txt_content_list[i])
            img_increase_list.append(img_list[i])
    return txt_content_list, img_class, img_increase, img_increase_list

if __name__ == "__main__":
    data_agumentation_path(['dataset_test/train/labels', 'dataset_test/valid/labels', 'dataset_test/test/labels'])
