from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import ndarray, expand_dims
from ...classes.training_config import TrainingConfig
from ...classes.image_augmentation_config import ImageAugmentationConfig

def get_train_generator(x_train: ndarray, y_train: ndarray, training_cfg: TrainingConfig, aug_cfg: ImageAugmentationConfig):
    
    datagen = ImageDataGenerator(
        rotation_range=aug_cfg.ROTATION_RANGE,
        width_shift_range=aug_cfg.WIDTH_SHIFT_RANGE,
        height_shift_range=aug_cfg.HEIGHT_SHIFT_RANGE,
        zoom_range=aug_cfg.ZOOM_RANGE,
        shear_range=aug_cfg.SHEAR_RANGE,
        horizontal_flip=aug_cfg.HORIZONTAL_FLIP,
        vertical_flip=aug_cfg.VERTICAL_FLIP,
        rescale=1./255,
    )
    
    train_generator=datagen.flow(
        expand_dims(x_train, axis=-1), 
        y_train,
        training_cfg.BATCH_SIZE
    )

    return train_generator