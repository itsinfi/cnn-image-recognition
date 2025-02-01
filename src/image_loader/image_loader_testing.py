import csv

import cv2 as cv

img = cv.imread('../../data/images/test.png', cv.IMREAD_GRAYSCALE)

cv.imshow('test', img)
flat_img = img.flatten()

print(flat_img)

with open('test.csv', 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(flat_img)

# k = cv.waitKey(0)