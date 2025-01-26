from .abstract.config import Config

class CompilerConfig(Config):

    LOSS: str
    OPTIMIZER: str
    METRICS: list[str]

    def __init__(self, LOSS: str, OPTIMIZER: str, METICS: list[str]) -> None:
        self.LOSS = LOSS
        self.OPTIMIZER = OPTIMIZER
        self.METRICS = METICS
        super().__init__()

    def __str__(self) -> str:
        repr = {
            'loss': self.LOSS,
            'optimizer': self.OPTIMIZER,
            'metrics': self.METRICS,
        }
        return str(repr)