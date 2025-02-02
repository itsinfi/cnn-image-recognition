from .abstract.config import Config

class LogConfig(Config):

    SHOW_FIRST_ENTRY: bool
    SHOW_SPLIT_DATA: bool
    SHOW_IMAGE_PLOT: bool
    SHOW_MODEL_PLOT: bool
    ENABLE_VERBOSE_TRAINING: bool

    def __init__(self, SHOW_FIRST_ENTRY: bool, SHOW_SPLIT_DATA: bool, SHOW_IMAGE_PLOT: bool, SHOW_MODEL_PLOT: bool, ENABLE_VERBOSE_TRAINING: bool) -> None:
        super().__init__()
        self.SHOW_FIRST_ENTRY = SHOW_FIRST_ENTRY
        self.SHOW_SPLIT_DATA = SHOW_SPLIT_DATA
        self.SHOW_IMAGE_PLOT = SHOW_IMAGE_PLOT
        self.SHOW_MODEL_PLOT = SHOW_MODEL_PLOT
        self.ENABLE_VERBOSE_TRAINING = ENABLE_VERBOSE_TRAINING

    def __str__(self) -> str:
        repr = {
            'SHOW_FIRST_ENTRY': self.SHOW_FIRST_ENTRY,
            'SHOW_SPLIT_DATA': self.SHOW_SPLIT_DATA,
            'SHOW_IMAGE_PLOT': self.SHOW_IMAGE_PLOT,
            'SHOW_MODEL_PLOT': self.SHOW_MODEL_PLOT,
            'ENABLE_VERBOSE_TRAINING': self.ENABLE_VERBOSE_TRAINING,
        }
        return str(repr)