import numpy as np
from src.model_generator.classes.image_data_config import ImageDataConfig
from src.model_generator.classes.log_config import LogConfig


def read_image_data(img_cfg: ImageDataConfig, log_cfg: LogConfig) -> list[np.ndarray[float]]:
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
            images.append(np.asarray(split_lines[1:], dtype='int').reshape(img_cfg.IMAGE_COMPR_SIZE))
    return images, labels