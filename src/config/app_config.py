from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D
from datetime import datetime
from src import CsvDataConfig
from src import TrainingConfig
from src import ModelConfig
from src import CompilerConfig
from src import ConvertToIntConfig
from src import JoinConfig
from src import LabelConfig
from src import ImageDataConfig
from src import LogConfig

SHOW_IMAGE_PLOT = False
SHOW_MODEL_PLOT = False

LOG_CFG: LogConfig = LogConfig(
    SHOW_FIRST_ENTRY=False,
    SHOW_IMAGE_PLOT=False,
    SHOW_SPLIT_DATA=False,
    SHOW_MODEL_PLOT=False,
)
CSV_CFG: list[CsvDataConfig] = [
]
IMAGE_CFG: ImageDataConfig = ImageDataConfig(
    DELIMITER=',',
    ROW_LIMIT=None,
    FILE_NAME='data/mnist_train.csv',
    FILTER_LIST=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    IMAGE_COMPR_SIZE=(28, 28),
)
LABEL_CFG: LabelConfig = LabelConfig(
    FILE='data/FuelConsumptionCo2.csv',
    NAME='CO2EMISSIONS',
)
TEST_SIZE: float = .45
MODEL_CFG: ModelConfig = ModelConfig(
    LAYERS=[
        Conv2D(filters=8, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 8)),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 8)),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 8)),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 8)),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 8)),
        Flatten(),
        Dense(units=16, activation='relu'),
        Dense(units=len(IMAGE_CFG.FILTER_LIST), activation='softmax'),
    ],
)
COMPILER_CFG: CompilerConfig = CompilerConfig(
    LOSS='sparse_categorical_crossentropy',
    OPTIMIZER='adam',
    METRICS=[
        'accuracy',
    ]
)
TRAINING_CFG: TrainingConfig = TrainingConfig(
    EPOCHS=3,
    BATCH_SIZE=32,
)