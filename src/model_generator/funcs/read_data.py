# from ..classes.csv_data_config import CsvDataConfig
from ..classes.image_data_config import ImageDataConfig
from ..classes.label_config import LabelConfig
from .utils import plot_images, read_image_data, split_image_data
from ..classes.log_config import LogConfig
# from pandas import read_csv

# TODO: return value
def read_data(img_cfg: ImageDataConfig, test_size: float, log_cfg: LogConfig):

    images, labels = read_image_data(img_cfg=img_cfg, log_cfg=log_cfg)
    
    if log_cfg.SHOW_IMAGE_PLOT:
        plot_images(images)

    return split_image_data(images=images, labels=labels, test_size=test_size, log_cfg=log_cfg)