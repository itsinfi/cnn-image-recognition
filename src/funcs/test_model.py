from ..classes.compiler_config import CompilerConfig
from tensorflow.keras import Sequential

def test_model(model: Sequential, x_test: any, y_test: any, compiler_cfg: CompilerConfig):
    loss, accuracy = model.evaluate(x_test, y_test)
    if 'accuracy' in compiler_cfg.METRICS:
        print('--- Accuracy:', accuracy)
    print('--- Loss:', loss)