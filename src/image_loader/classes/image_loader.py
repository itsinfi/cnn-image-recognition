import csv
import cv2 as cv
import numpy

class ImageLoader:
    def __init__(self):
        pass

    def load_image(self, image_name: str):

        img = cv.imread('data/images/'+image_name, cv.IMREAD_GRAYSCALE)
        img = img.flatten()
        label = [0]
        # fixme: remove debug
        print('label: '+str(label))
        print('img: '+ img)
        img = label + img
        print('label + img: '+ img)
        return img

    def export_list_to_csv(self, img_list: list[numpy.ndarray], csv_file_name):
        with open('data/' + csv_file_name, 'w') as f:
            csv_writer = csv.writer(f)
            for img in img_list:
                csv_writer.writerow(img)

    def export_to_csv(self, img: numpy.ndarray, csv_file_name: str):
        with open('data/'+csv_file_name, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(img)