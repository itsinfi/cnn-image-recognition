import numpy as np
from src.classes.image_data_config import ImageDataConfig


def read_image_data(img_cfg: ImageDataConfig) -> list[np.ndarray[float]]:
    # read image data
    csv = open(img_cfg.FILE_NAME, 'r')
    lines = csv.readlines()
    csv.close()

    # display first number
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
        if split_lines[0] in img_cfg.FILTER_LIST:
            labels.append(float(split_lines[0]))
            images.append(np.asarray(split_lines[1:], dtype='float').reshape((28,28)))
    return images, labels