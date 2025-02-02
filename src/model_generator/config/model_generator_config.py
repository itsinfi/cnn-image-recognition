from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from src import CsvDataConfig
from src import TrainingConfig
from src import ModelConfig
from src import CompilerConfig
from src import ImageDataConfig
from src import LogConfig
from src import ImageAugmentationConfig
from src import EarlyStoppingConfig

MODEL_NAME = 'mnist'

SHOW_IMAGE_PLOT = False
SHOW_MODEL_PLOT = False

LOG_CFG: LogConfig = LogConfig(
    ENABLE_VERBOSE_TRAINING=True,
    SHOW_FIRST_ENTRY=False,
    SHOW_IMAGE_PLOT=False,
    SHOW_SPLIT_DATA=False,
    SHOW_MODEL_PLOT=True,
)
IMAGE_CFG: ImageDataConfig = ImageDataConfig(
    DELIMITER=',',
    ROW_LIMIT=None,
    FILE_NAME='data/mnist_train.csv',
    FILTER_LIST=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    IMAGE_COMPR_SIZE=(28, 28),
)
TEST_SIZE: float = .2
MODEL_CFG: ModelConfig = ModelConfig(
    LAYERS=[
        # Input layer
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        # 1st hidden layer
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 2nd hiden layer
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),

        # Last hidden layer
        Flatten(),
        Dense(units=128, activation='relu'),
        Dropout(rate=0.5),

        # Output layer
        Dense(units=10, activation='softmax'),
    ],
)
COMPILER_CFG: CompilerConfig = CompilerConfig(
    LOSS=SparseCategoricalCrossentropy(),
    OPTIMIZER=Adam(learning_rate=0.0005),
    METRICS=[
        'accuracy',
    ]
)
TRAINING_CFG: TrainingConfig = TrainingConfig(
    EPOCHS=50,
    BATCH_SIZE=128,
    STEPS_PER_EPOCH= (0.8 * 57000) // 128,
    VERBOSE=2,
)

AUG_CFG: ImageAugmentationConfig = ImageAugmentationConfig(
    ENABLED=True,
    ROTATION_RANGE=10,
    WIDTH_SHIFT_RANGE=0.05,
    HEIGHT_SHIFT_RANGE=0.05,
    ZOOM_RANGE=0.05,
    SHEAR_RANGE=0.05,
    HORIZONTAL_FLIP=False,
    VERTICAL_FLIP=False,
)

ES_CFG: EarlyStoppingConfig = EarlyStoppingConfig(
    ENABLED=True,
    MONITOR='loss',
    PATIENCE=2,
    RESTORE_BEST_WEIGHTS=True,
)