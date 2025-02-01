from ...classes.csv_data_config import CsvDataConfig
from pandas import to_datetime, DataFrame
from datetime import datetime

def apply_int_conversions(csv_cfg: CsvDataConfig, csv: DataFrame) -> DataFrame:
    for to_int_cfg in csv_cfg.CONVERT_TO_INT:
            val = to_datetime(csv[to_int_cfg.COL])
            if to_int_cfg.TYPE is datetime:
                csv['year'] = val.dt.year
                csv['month'] = val.dt.month
                csv['day'] = val.dt.day
            csv = csv.drop(columns=[to_int_cfg.COL])
    return csv