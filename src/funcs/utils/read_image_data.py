import numpy as np
from src.classes.image_data_config import ImageDataConfig


def read_image_data(img_cfg: ImageDataConfig) -> list[np.ndarray[float]]:
    # read image data
    data_file = open(img_cfg.FILE_NAME, 'r')
    data_lines = data_file.readlines()
    data_file.close()

    # display first number
    pic_values = data_lines[0].split(img_cfg.DELIMITER)
    pic_array = np.asarray(pic_values[1:], dtype='float').reshape((28, 28))
    for z in range(28):
        for s in range(28):
            print(str(int(pic_array[z][s])).rjust(3), end=' ')
        print()
    
    # read row limit
    if img_cfg.ROW_LIMIT is None:
        limit = len(data_lines)
    else: 
        limit = img_cfg.ROW_LIMIT

    # create a list of image data
    images = []
    for i in range(limit):
        pics_values = data_lines[i].split(img_cfg.DELIMITER)
        images.append(np.asarray(pics_values[1:], dtype='float').reshape((28,28)))
    return images