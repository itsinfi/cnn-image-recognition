from tensorflow.keras.layers import Dense, Input
from datetime import datetime
from src import CsvDataConfig
from src import TrainingConfig
from src import ModelConfig
from src import CompilerConfig
from src import ConvertToIntConfig
from src import JoinConfig
from src import LabelConfig

CSV_CFG: list[CsvDataConfig] = [
    CsvDataConfig(
        FILE_NAME='data/FuelConsumptionCo2.csv',
        DELIMITER=',',
        ROW_LIMIT=None,
        SELECTED_COLS=[
            'CO2EMISSIONS','FUELCONSUMPTION_COMB_MPG',
        ],
        CONVERT_TO_INT=[],
        CONVERT_TO_LIST=[]
    )
]
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