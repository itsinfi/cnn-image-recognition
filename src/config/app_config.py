from tensorflow.keras.layers import Dense, Input
from datetime import datetime
from src import CsvDataConfig
from src import TrainingConfig
from src import ModelConfig
from src import CompilerConfig
from src import ConvertToIntConfig
from src import JoinConfig
from src import LabelConfig
from src.classes import ImageDataConfig

CSV_CFG: list[CsvDataConfig] = [

]

IMAGE_CFG = ImageDataConfig(
    DELIMITER=',',
    ROW_LIMIT=10,
    FILE_NAME='data/mnist_train_100.csv'

)

LABEL_CFG = LabelConfig(
    FILE='data/FuelConsumptionCo2.csv',
    NAME='CO2EMISSIONS',
)
TEST_SIZE = .2
MODEL_CFG: ModelConfig = ModelConfig(
    LAYERS=[
        Input(shape=(1,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1),
    ],
)
COMPILER_CFG: CompilerConfig = CompilerConfig(
    LOSS='mse',
    OPTIMIZER='adam',
    METRICS=[
        'mae',
    ]
)
TRAINING_CFG: TrainingConfig = TrainingConfig(
    EPOCHS=50,
    BATCH_SIZE=32,
)