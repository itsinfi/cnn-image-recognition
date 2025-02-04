import numpy as np
import os
from src.model_generator.classes.image_data_config import ImageDataConfig
from src.model_generator.classes.log_config import LogConfig
from src.image_loader.classes.image_loader import ImageLoader
from src.model_generator.config.model_generator_config import IMAGE_CFG

def read_image_data(img_cfg: ImageDataConfig, log_cfg: LogConfig) -> list[np.ndarray[float]]:

    ROW_LIMIT = img_cfg.ROW_LIMIT

    total_count = 0

    il = ImageLoader()
    # load files into a list
    print('loading image files into a list')
    img_file_list = []
    for img_file in os.listdir('data/custom_test_2'):
        if ROW_LIMIT != None and ROW_LIMIT == 0:
            break
        if img_file.endswith('.jpg'):
            img_file_list.append(img_file)
            total_count += 1
            if ROW_LIMIT != None:
                ROW_LIMIT -= 1
    
    # read images into a list
    print('reading images')
    img_list = []
    label_list = []
    file_name_list = []
    for img_file in img_file_list:
        print('loading: ', img_file)
        img, label, file_name = il.load_image(image_name=img_file, target_size=IMAGE_CFG.IMAGE_SIZE)

        img_list.append(img)
        label_list.append(label)
        file_name_list.append(file_name)
    return img_list, label_list, file_name_list

def read_image_data_from_csv(img_cfg: ImageDataConfig, log_cfg: LogConfig) -> list[np.ndarray[float]]:
    # read image data
    csv = open(img_cfg.FILE_NAME, 'r')
    lines = csv.readlines()
    csv.close()

    # display first number
    if log_cfg.SHOW_FIRST_ENTRY:
        pic_values = lines[0].split(img_cfg.DELIMITER)
        pic_array = np.asarray(pic_values[1:], dtype='float').reshape((28, 28))
        for z in range(28):
            for s in range(28):
                print(str(int(pic_array[z][s])).rjust(3), end=' ')
            print()
    
    # read row limit
    if img_cfg.ROW_LIMIT is None:
        limit = len(lines)
    else: 
        limit = img_cfg.ROW_LIMIT

    # create a list of image data
    images = []
    labels = []
    for i in range(limit):
        split_lines = lines[i].split(img_cfg.DELIMITER)
        # filter
        if len(img_cfg.FILTER_LIST) == 0 or split_lines[0] in img_cfg.FILTER_LIST:
            labels.append(int(split_lines[0]))
            images.append(np.asarray(split_lines[1:], dtype='int').reshape(img_cfg.IMAGE_SIZE))
    return images, labels