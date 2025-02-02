from .abstract.config import Config

class ImageAugmentationConfig(Config):

    ENABLED: bool
    ROTATION_RANGE: int
    WIDTH_SHIFT_RANGE: float
    HEIGHT_SHIFT_RANGE: float
    ZOOM_RANGE: float
    SHEAR_RANGE: float
    HORIZONTAL_FLIP: bool
    VERTICAL_FLIP: bool

    def __init__(
        self, 
        ENABLED: bool,
        ROTATION_RANGE: int,
        WIDTH_SHIFT_RANGE: float,
        HEIGHT_SHIFT_RANGE: float,
        ZOOM_RANGE: float,
        SHEAR_RANGE: float,
        HORIZONTAL_FLIP: bool,
        VERTICAL_FLIP: bool
    ):
        super().__init__()
        self.ENABLED = ENABLED
        self.ROTATION_RANGE=ROTATION_RANGE
        self.WIDTH_SHIFT_RANGE=WIDTH_SHIFT_RANGE
        self.HEIGHT_SHIFT_RANGE=HEIGHT_SHIFT_RANGE
        self.ZOOM_RANGE=ZOOM_RANGE
        self.SHEAR_RANGE=SHEAR_RANGE
        self.HORIZONTAL_FLIP=HORIZONTAL_FLIP
        self.VERTICAL_FLIP=VERTICAL_FLIP

    def __str__(self):
        repr = {
            'ENABLED': self.ENABLED,
            'ROTATION_RANGE': self.ROTATION_RANGE,
            'WIDTH_SHIFT_RANGE': self.WIDTH_SHIFT_RANGE,
            'HEIGHT_SHIFT_RANGE': self.HEIGHT_SHIFT_RANGE,
            'ZOOM_RANGE': self.ZOOM_RANGE,
            'SHEAR_RANGE': self.SHEAR_RANGE,
            'HORIZONTAL_FLIP': self.HORIZONTAL_FLIP,
            'VERTICAL_FLIP': self.VERTICAL_FLIP,
        }
        return str(repr)