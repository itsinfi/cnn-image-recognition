from pandas import get_dummies, DataFrame
from sklearn.model_selection import train_test_split
from numpy import int32
from ...classes.log_config import LogConfig

def split_data(joined_csv: DataFrame, label_name: str, test_size: float, log_cfg:LogConfig) -> tuple[any]:
    # split dependent and independent variables
    x = get_dummies(joined_csv.drop([label_name], axis=1)).to_numpy(dtype=int32)
    y = joined_csv[label_name].to_numpy(dtype=int32)

    if log_cfg.SHOW_SPLIT_DATA:
        print('x:', x)
        print('y:', y)

    # split training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    if log_cfg.SHOW_SPLIT_DATA:
        print(x_test)
    return x_train, x_test, y_train, y_test