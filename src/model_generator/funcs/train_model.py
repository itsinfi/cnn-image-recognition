from ..classes.model_config import ModelConfig
from ..classes.compiler_config import CompilerConfig
from ..classes.training_config import TrainingConfig
from tensorflow.keras import Sequential
from numpy import ndarray
from .utils.plot_model import plot_model

def train_model(x_train: ndarray, y_train: ndarray, model_cfg: ModelConfig, compiler_cfg: CompilerConfig, training_cfg: TrainingConfig, show_model_plot: bool) -> Sequential:
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

    if show_model_plot:
        plot_model(model)

    return model