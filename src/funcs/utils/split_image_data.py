from sklearn.model_selection import train_test_split
from numpy import ndarray, array

def split_image_data(images: list[ndarray[float]], labels: list[float], test_size: float) -> tuple[ndarray[int]]:

    x = array(images)
    y = array(labels)

    print('x:', x)
    print('y:', y)

    # split training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test