# from ..classes.csv_data_config import CsvDataConfig
from ..classes.image_data_config import ImageDataConfig
from ..classes.label_config import LabelConfig
from .utils import plot_images, read_image_data, split_image_data
from ..classes.log_config import LogConfig
# from pandas import read_csv

# TODO: return value
def read_data(img_cfg: ImageDataConfig, label_cfg: LabelConfig, test_size: float, log_cfg: LogConfig):
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


    # Join tables TODO: create add a proper join config + think about join types
    # joined_csv = join_data(csv_files=csv_files)

    # split data
    # return split_data(joined_csv=data_lines, label_name=label_cfg.NAME, test_size=test_size)

    images, labels = read_image_data(img_cfg=img_cfg, log_cfg=log_cfg)
    
    if log_cfg.SHOW_IMAGE_PLOT:
        plot_images(images)

    return split_image_data(images=images, labels=labels, test_size=test_size, log_cfg=log_cfg)