from .abstract.config import Config

class EarlyStoppingConfig(Config):
    
    ENABLED: bool
    MONITOR: str
    PATIENCE: int
    RESTORE_BEST_WEIGHTS: bool

    def __init__(self, ENABLED: bool, MONITOR: str, PATIENCE: int, RESTORE_BEST_WEIGHTS: bool):
        super().__init__()
        self.ENABLED = ENABLED
        self.MONITOR = MONITOR
        self.PATIENCE = PATIENCE
        self.RESTORE_BEST_WEIGHTS = RESTORE_BEST_WEIGHTS

    def __str__(self):
        repr = {
            'ENABLED': self.ENABLED,
            'MONITOR': self.MONITOR,
            'PATIENCE': self.PATIENCE,
            'RESTORE_BEST_WEIGHTS': self.RESTORE_BEST_WEIGHTS,
        }
        return str(repr)

