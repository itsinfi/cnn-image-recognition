import csv
import cv2 as cv
import numpy
import re

class ImageLoader:
    def __init__(self):
        pass

    def load_image(self, image_name: str, target_size: tuple[int]) -> numpy.ndarray:
        img = cv.imread('data/train/'+image_name, cv.IMREAD_GRAYSCALE)

        img = cv.resize(img, target_size, interpolation=cv.INTER_LINEAR)

        img = img.flatten().tolist()

        label = re.search('\w+(?=\.)', image_name)

        match label.group():
            case 'cat':
                label = [0]
            case 'dog':
                label = [1]

        img = label + img
        return numpy.array(img)

    def export_list_to_csv(self, img_list: list[numpy.ndarray], csv_file_name):
        with open('data/' + csv_file_name, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(img_list)

    def export_to_csv(self, img: numpy.ndarray, csv_file_name: str):
        with open('data/'+csv_file_name, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(img)