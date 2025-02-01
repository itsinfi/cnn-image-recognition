import csv
import cv2 as cv
import numpy

class ImageLoader:
    def __init__(self):
        pass

    def load_image(self, image_name: str):

        img = cv.imread('data/images/'+image_name, cv.IMREAD_GRAYSCALE)

        return img

    def export_to_csv(self, img: numpy.ndarray, csv_file_name: str):
        img = img.flatten()
        with open('data/'+csv_file_name, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(img)