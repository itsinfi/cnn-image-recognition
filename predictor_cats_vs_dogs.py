# from src.image_loader.classes.image_loader import ImageLoader
# import os
from src.model_generator.funcs.utils.read_image_data import read_image_data
from src.image_loader.funcs.utils.load_model import load_model
from src import ImageDataConfig
from src.model_generator.config import *
from numpy import array, argmax, asarray, unique
import numpy as np

img_cfg = ImageDataConfig(
    DELIMITER=',',
    ROW_LIMIT=None,
    FILE_NAME='data/exported.csv',
    FILTER_LIST=['0', '1'],
    IMAGE_SIZE=(150, 150),
)

images, labels = read_image_data(img_cfg=img_cfg, log_cfg=LOG_CFG)

model = load_model('cats_vs_dogs', 'cats_vs_dogs_different_parameters_Acc93-Loss0,16')

predictions = model.predict(array(images))

l = 0

print(f'Einzigartige Labels in den Trainingsdaten: {model.output_shape[-1]}')

for img, pred in zip(images, predictions):
    pic_array = asarray(img, dtype='float')
    print('Prediction:', argmax(pred), 'actual label:', labels[l], '\n')
    print(pred)
    l+=1