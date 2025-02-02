from src.image_loader.classes.image_loader import ImageLoader
import os
from src.model_generator.funcs.utils.read_image_data import read_image_data
from src.image_loader.funcs.utils.load_model import load_model
from src import ImageDataConfig
from src.model_generator.config import *
from numpy import array, argmax, asarray

print('start')
il = ImageLoader()

# load files into a list
print('loading image files into a list')
img_file_list = []
for img_file in os.listdir('data/train'):
    if img_file.endswith('.jpg'):
        img_file_list.append(img_file)

# read images into a list
print('reading images')
img_list = []
for img_file in img_file_list:
    print('loading: ', img_file)
    img_list.append(il.load_image(img_file))

# export images from a list to a csv
print('exporting images to csv')
il.export_list_to_csv(img_list, 'exported.csv')
exit()

# read csvs
img_cfg = ImageDataConfig(
    DELIMITER=',',
    ROW_LIMIT=None,
    FILE_NAME='data/exported.csv',
    FILTER_LIST=['0', '1'],
    IMAGE_COMPR_SIZE=(28, 28),
)
data, labels = read_image_data(img_cfg=img_cfg, log_cfg=LOG_CFG)

model = load_model(MODEL_NAME)

predictions = model.predict(array(data))

for dick, pred in zip(data, predictions):
    pic_array = asarray(dick, dtype='float')
    for z in range(28):
        for s in range(28):
            print(str(int(pic_array[z][s])).rjust(3), end=' ')
        print()
    print('LABEL:', argmax(pred), '\n')