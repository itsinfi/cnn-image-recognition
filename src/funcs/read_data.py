import numpy as np
import matplotlib as plt

from ..classes.csv_data_config import CsvDataConfig
from ..classes.image_data_config import ImageDataConfig
from ..classes.label_config import LabelConfig
from .utils import apply_int_conversions, apply_list_conversions, join_data, split_data
from pandas import read_csv

# TODO: return value
def read_data(csv_cfg: ImageDataConfig, label_cfg: LabelConfig, test_size: float):
    # csv_files = []
    #
    # # for all csv configs
    # for csv_cfg in csv_cfg_list:
    #
    #     # read csv
    #     csv = read_csv(
    #         csv_cfg.FILE_NAME,
    #         delimiter=csv_cfg.DELIMITER,
    #         nrows=csv_cfg.ROW_LIMIT,
    #         usecols=csv_cfg.SELECTED_COLS
    #     )
    #
    #     # apply conversions to int (if specified)
    #     csv = apply_int_conversions(csv_cfg=csv_cfg, csv=csv)
    #
    #     # apply list conversion (if specified)
    #     csv = apply_list_conversions(csv_cfg=csv_cfg, csv=csv)
    #
    #     # attach to list
    #     csv_files.append(csv)

    data_file = open("data/mnist_train_100.csv", 'r')
    data_lines = data_file.readlines()
    data_file.close()
    pic_array = []
    for pic_nr in range(len(data_lines)):
        pic_values = data_lines[pic_nr].split(',')
        pic_array.append(pic_values)

    pic_array = np.asarray(pic_values[1:], dtype='float').reshape((28, 28))

    # ASCII-Ausgabe der Bilddaten

    for z in range(28):

        for s in range(28):
            # Ausgabe der Werte mit 3 Stellen, rechtsbündig

            print(str(int(pic_array[z][s])).rjust(3), end=' ')

        print()


    # read image data

    # Join tables TODO: create add a proper join config + think about join types
    # joined_csv = join_data(csv_files=csv_files)

    # split data
    # return split_data(joined_csv=data_lines, label_name=label_cfg.NAME, test_size=test_size)
    plot_pics(data_lines)


def plot_pics(data_lines):
    images=[]
    for i in range(len(data_lines)):
        pics_values = data_lines[i].split(',')
        images.append(np.asarray(pics_values[1:], dtype='float').reshape((28,28)))
    fig, axes = plt.subplots(nrows=1, ncols=len(images))
    for j, ax in enumerate(axes):
        ax.matshow(images[j].reshape(28,28), cmap = plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()