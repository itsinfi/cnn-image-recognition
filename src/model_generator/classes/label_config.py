from .abstract.config import Config

class LabelConfig(Config):

    NAME: str
    FILE: str

    def __init__(self, NAME: str, FILE: str):
        self.NAME = NAME
        self.FILE = FILE
        super().__init__()

    def __str__(self):
        repr = {
            'name': self.NAME,
            'file': self.FILE,
        }
        return str(repr)