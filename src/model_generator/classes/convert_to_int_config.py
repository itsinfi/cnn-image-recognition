from .abstract.config import Config
from typing import Type

class ConvertToIntConfig(Config):
    COL: str
    TYPE: Type
    FORMAT: str

    def __init__(self, COL: str, TYPE: Type, FORMAT: str) -> None:
        self.COL = COL
        self.TYPE = TYPE
        self.FORMAT = FORMAT
        super().__init__()

    def __str__(self) -> str:
        repr = {
            'col': self.COL,
            'type': self.TYPE,
            'format': self.FORMAT,
        }
        return str(repr)