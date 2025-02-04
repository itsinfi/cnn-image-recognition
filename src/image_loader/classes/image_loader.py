import csv
import cv2 as cv
import numpy
import re
from tensorflow import convert_to_tensor, float32

class ImageLoader:
    def __init__(self):
        pass

    def load_image(self, image_name: str, target_size: tuple[int]) -> numpy.ndarray:
        img_file = cv.imread('data/custom_test_2/'+image_name)

        img_resize = cv.resize(img_file, target_size, interpolation=cv.INTER_LINEAR)

        img_rgb = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)

        img_tensor = convert_to_tensor(img_rgb, dtype=float32)

        img = img_tensor / 255.0

        label = re.search('\w+(?=\.)', image_name)

        match label.group():
            case 'cat':
                label = [0]
            case 'dog':
                label = [1]
            case _:
                label = [3]

        return img, label, image_name

    def export_list_to_csv(self, img_list: list[numpy.ndarray], csv_file_name):
        with open('data/' + csv_file_name, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(img_list)

    def export_to_csv(self, img: numpy.ndarray, csv_file_name: str):
        with open('data/'+csv_file_name, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(img)