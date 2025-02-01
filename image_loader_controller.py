from src.image_loader.classes.image_loader import ImageLoader

il = ImageLoader()

img = il.load_image('test.png')

il.export_to_csv(img, 'test.csv')