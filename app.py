from src.model_generator.config import *
from src import read_data
from src import train_model
from src import test_model

x_train, x_test, y_train, y_test = read_data(
    img_cfg=IMAGE_CFG,
    label_cfg=LABEL_CFG, 
    test_size=TEST_SIZE,
    log_cfg=LOG_CFG,
)

model = train_model(
    x_train=x_train,
    y_train=y_train, 
    model_cfg=MODEL_CFG, 
    compiler_cfg=COMPILER_CFG, 
    training_cfg=TRAINING_CFG,
    log_cfg=LOG_CFG,
)

test_model(
    model=model,
    x_test=x_test,
    y_test=y_test,
    compiler_cfg=COMPILER_CFG
)

model.save(f'{MODEL_NAME}.h5')