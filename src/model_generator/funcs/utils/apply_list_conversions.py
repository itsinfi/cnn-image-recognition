from ...classes.csv_data_config import CsvDataConfig
from pandas import DataFrame
from .parse_list import parse_list

def apply_list_conversions(csv_cfg: CsvDataConfig, csv: DataFrame) -> DataFrame:
    for key in csv_cfg.CONVERT_TO_LIST:
        csv[key] = csv[key].apply(parse_list)
        csv = csv.explode(key, ignore_index=True)
    return csv