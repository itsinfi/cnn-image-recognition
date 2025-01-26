from .abstract.config import Config
from tensorflow.keras.layers import Dense

class ModelConfig(Config):

    LAYERS: list[Dense]

    def __init__(self, LAYERS: list[Dense]) -> None:
        self.LAYERS = LAYERS
        super().__init__()

    def __str__(self) -> str:
        repr = {
            'layers': self.LAYERS,
        }
        return str(repr)