from .abstract.config import Config

class TrainingConfig(Config):

    EPOCHS: int
    BATCH_SIZE: int

    def __init__(self, EPOCHS: int, BATCH_SIZE: int) -> None:
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        super().__init__()

    def __str__(self) -> str:
        repr = {
            'epochs': self.EPOCHS,
            'batch_size': self.BATCH_SIZE,
        }
        return str(repr)