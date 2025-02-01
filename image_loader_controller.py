from src.image_loader.classes.image_loader import ImageLoader
import os

il = ImageLoader()

# load files into a list
img_file_list = []
for img_file in os.listdir('data/images'):
    if img_file.endswith('.png'):
        img_file_list.append(img_file)
# read images into a list
img_list = []
for img_file in img_file_list:
    img_list.append(il.load_image(img_file))

# export images from a list to a csv
il.export_list_to_csv(img_list, 'exported.csv')