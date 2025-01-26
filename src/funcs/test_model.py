from ..classes.compiler_config import CompilerConfig
from tensorflow.keras import Sequential
from numpy import ndarray

def test_model(model: Sequential, x_test: ndarray, y_test: ndarray, compiler_cfg: CompilerConfig):
    loss, metric = model.evaluate(x_test, y_test)
    if 'accuracy' in compiler_cfg.METRICS:
        print('--- Accuracy:', metric)
    if 'mae' in compiler_cfg.METRICS:
        print('--- MAE:', metric)
    print('--- Loss:', loss)