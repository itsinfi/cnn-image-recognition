from .abstract.config import Config

class TrainingConfig(Config):

    EPOCHS: int
    BATCH_SIZE: int
    STEPS_PER_EPOCH: int
    VERBOSE: int

    def __init__(self, EPOCHS: int, BATCH_SIZE: int, STEPS_PER_EPOCH: int, VERBOSE: int) -> None:
        super().__init__()
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.STEPS_PER_EPOCH = STEPS_PER_EPOCH
        self.VERBOSE = VERBOSE

    def __str__(self) -> str:
        repr = {
            'EPOCHS': self.EPOCHS,
            'BATCH_SIZE': self.BATCH_SIZE,
            'STEPS_PER_EPOCH': self.STEPS_PER_EPOCH,
            'VERBOSE': self.VERBOSE,
        }
        return str(repr)