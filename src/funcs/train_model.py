from ..classes.model_config import ModelConfig
from ..classes.compiler_config import CompilerConfig
from ..classes.training_config import TrainingConfig
from tensorflow.keras import Sequential
from numpy import ndarray
from .utils.plot_model import plot_model
from ..classes.log_config import LogConfig

def train_model(x_train: ndarray, y_train: ndarray, model_cfg: ModelConfig, compiler_cfg: CompilerConfig, training_cfg: TrainingConfig, log_cfg: LogConfig) -> Sequential:
    model = Sequential(
        model_cfg.LAYERS
    )

    model.compile(
        loss=compiler_cfg.LOSS,
        optimizer=compiler_cfg.OPTIMIZER,
        metrics=compiler_cfg.METRICS,
    )

    model.fit(
        x_train,
        y_train,
        epochs=training_cfg.EPOCHS,
        batch_size=training_cfg.BATCH_SIZE,
    )

    if log_cfg.SHOW_MODEL_PLOT:
        plot_model(model)

    return model