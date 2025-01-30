from sklearn.model_selection import train_test_split
from numpy import ndarray, array
from ...classes.log_config import LogConfig

def split_image_data(images: list[ndarray[float]], labels: list[float], test_size: float, log_cfg: LogConfig) -> tuple[ndarray[int]]:

    x = array(images)
    y = array(labels)

    if log_cfg.SHOW_SPLIT_DATA:
        print('x:', x)
        print('y:', y)

    # split training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    if log_cfg.SHOW_SPLIT_DATA:
        print(f'x_train: {len(x_train)}, x_test: {len(x_test)}, y_train: {len(y_train)}, y_test: {len(y_test)}')
    return x_train, x_test, y_train, y_test