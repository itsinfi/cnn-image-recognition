from ..classes.model_config import ModelConfig
from ..classes.compiler_config import CompilerConfig
from ..classes.training_config import TrainingConfig
from tensorflow.keras import Sequential

def train_model(x_train: any, y_train: any, model_cfg: ModelConfig, compiler_cfg: CompilerConfig, training_cfg: TrainingConfig) -> Sequential:
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

    return model