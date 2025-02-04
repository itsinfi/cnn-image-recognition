import csv
import cv2 as cv
import numpy

class ImageLoader:
    def __init__(self):
        pass

    def load_image(self, image_name: str) -> numpy.ndarray:
        img = cv.imread('data/images/'+image_name, cv.IMREAD_GRAYSCALE)
        img = img.flatten().tolist()
        label = [9]
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