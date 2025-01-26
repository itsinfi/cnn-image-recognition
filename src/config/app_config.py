from tensorflow.keras.layers import Dense
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
            'CO2EMISSIONS','FUELCONSUMPTION_COMB',
        ],
        CONVERT_TO_INT=[],
        CONVERT_TO_LIST=[]
    )
    # CsvDataConfig(
    #     FILE_NAME='data/category_hierarchy.csv',
    #     DELIMITER='|',
    #     ROW_LIMIT=5000,
    #     SELECTED_COLS=[
    #         'category',
    #         'parent_category',
    #     ],
    #     CONVERT_TO_INT=[],
    #     CONVERT_TO_LIST=[],
    # ),
    # CsvDataConfig(
    #     FILE_NAME='data/items.csv',
    #     DELIMITER='|',
    #     ROW_LIMIT=5000,
    #     SELECTED_COLS=[
    #         'itemID',
    #         'brand',
    #         'feature_1',
    #         'feature_2',
    #         'feature_3',
    #         'feature_4',
    #         'feature_5',
    #         'categories',
    #     ],
    #     CONVERT_TO_INT=[],
    #     CONVERT_TO_LIST=[
    #         'categories'
    #     ],
    # ),
    # CsvDataConfig(
    #     FILE_NAME='data/orders.csv',
    #     DELIMITER='|',
    #     ROW_LIMIT=5000,
    #     SELECTED_COLS=[
    #         'date',
    #         'userID',
    #         'itemID',
    #         'order',
    #     ],
    #     CONVERT_TO_INT=[
    #         ConvertToIntConfig(
    #             COL='date',
    #             TYPE=datetime,
    #             FORMAT='%Y-%m-%d'
    #         ),
    #     ],
    #     CONVERT_TO_LIST=[],
    # ),
    # CsvDataConfig(
    #     FILE_NAME='data/submission.csv',
    #     DELIMITER='|',
    #     ROW_LIMIT=5000,
    #     SELECTED_COLS=[
    #         'userID',
    #         'itemID',
    #     ],
    #     CONVERT_TO_INT=[],
    #     CONVERT_TO_LIST=[],
    # ),
]
LABEL_CFG = LabelConfig(
    FILE='data/FuelConsumptionCo2.csv',
    NAME='CO2EMISSIONS',
)
TEST_SIZE = .2
MODEL_CFG: ModelConfig = ModelConfig(
    LAYERS=[
        Dense(units=2, activation='relu'),
        Dense(units=1, activation='softmax'),
    ],
)
COMPILER_CFG: CompilerConfig = CompilerConfig(
    LOSS='mean_squared_error',
    OPTIMIZER='adam',
    METICS=[
        'mae',
    ]
)
TRAINING_CFG: TrainingConfig = TrainingConfig(
    EPOCHS=50,
    BATCH_SIZE=32,
)