from ..classes.csv_data_config import CsvDataConfig
from ..classes.label_config import LabelConfig
from .utils import apply_int_conversions, apply_list_conversions, join_data, split_data
from pandas import read_csv

# TODO: return value
def read_data(csv_cfg_list: list[CsvDataConfig], label_cfg: LabelConfig, test_size: float):
    csv_files = []
    
    # for all csv configs
    for csv_cfg in csv_cfg_list:

        # read csv
        csv = read_csv(
            csv_cfg.FILE_NAME, 
            delimiter=csv_cfg.DELIMITER,
            nrows=csv_cfg.ROW_LIMIT,
            usecols=csv_cfg.SELECTED_COLS
        )

        # apply conversions to int (if specified)
        csv = apply_int_conversions(csv_cfg=csv_cfg, csv=csv)

        # apply list conversion (if specified)
        csv = apply_list_conversions(csv_cfg=csv_cfg, csv=csv)

        # attach to list
        csv_files.append(csv)

    # Join tables TODO: create add a proper join config + think about join types
    joined_csv = join_data(csv_files=csv_files)

    # split data
    return split_data(joined_csv=joined_csv, label_name=label_cfg.NAME, test_size=test_size)
