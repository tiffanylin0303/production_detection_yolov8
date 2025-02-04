import DetectClass

path = ['dataset_test/train/labels', 'dataset_test/valid/labels', 'dataset_test/test/labels']
txt_content_list, img_class, img_increase, img_increase_list = DetectClass.data_augmentation_path(path)