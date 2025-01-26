from ..convert_to_int_config import ConvertToIntConfig
from .config import Config

class DataConfig(Config):

    FILE_NAME: str
    SELECTED_COLS: list[str] 
    ROW_LIMIT: int
    CONVERT_TO_INT: list[ConvertToIntConfig]

    def __init__(
        self, 
        FILE_NAME: str,
        SELECTED_COLS: list[str], 
        ROW_LIMIT: int, 
        CONVERT_TO_INT: list[ConvertToIntConfig]
    ) -> None:
        self.FILE_NAME = FILE_NAME
        self.SELECTED_COLS = SELECTED_COLS
        self.ROW_LIMIT = ROW_LIMIT
        self.CONVERT_TO_INT = CONVERT_TO_INT
        super().__init__()


    def __str__(self) -> str:
        repr = {
            'file_name': self.FILE_NAME,
            'selected_cols': self.SELECTED_COLS,
            'limit': self.ROW_LIMIT,
            'convert_to_int': self.CONVERT_TO_INT,
        }
        return str(repr)