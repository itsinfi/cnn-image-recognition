from ..classes.model_config import ModelConfig
from ..classes.compiler_config import CompilerConfig
from ..classes.training_config import TrainingConfig
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from numpy import ndarray
from .utils.plot_model import plot_model
from ..classes.log_config import LogConfig
from ..classes.image_augmentation_config import ImageAugmentationConfig
from ..classes.early_stopping_config import EarlyStoppingConfig
from .utils.get_train_generator import get_train_generator
from .utils.fit_model import fit_model

def train_model(
        model_name: str,
        x_train: ndarray,
        y_train: ndarray, 
        model_cfg: ModelConfig,
        compiler_cfg: CompilerConfig,
        training_cfg: TrainingConfig,
        log_cfg: LogConfig,
        aug_cfg: ImageAugmentationConfig,
        es_cfg: EarlyStoppingConfig,
    ) -> Sequential:

    model = Sequential(
        model_cfg.LAYERS
    )

    model.compile(
        loss=compiler_cfg.LOSS,
        optimizer=compiler_cfg.OPTIMIZER,
        metrics=compiler_cfg.METRICS,
    )

    callbacks = []
    train_generator = None
    
    if aug_cfg.ENABLED:
        train_generator = get_train_generator(
            x_train=x_train,
            y_train=y_train,
            training_cfg=training_cfg,
            aug_cfg=aug_cfg,
        )
    
    if es_cfg.ENABLED:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.MONITOR,
                patience=es_cfg.PATIENCE,
                restore_best_weights=es_cfg.RESTORE_BEST_WEIGHTS,
            )
        )

    fit_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        train_generator=train_generator,
        steps_per_epoch=training_cfg.STEPS_PER_EPOCH,
        epochs=training_cfg.EPOCHS,
        verbose= 1 if log_cfg.ENABLE_VERBOSE_TRAINING else 0,
        callbacks=callbacks
    )

    if log_cfg.SHOW_MODEL_PLOT:
        plot_model(model=model, model_name=model_name)

    return model