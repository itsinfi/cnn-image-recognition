from .convert_to_int_config import ConvertToIntConfig
from .abstract.data_config import DataConfig

class ImageDataConfig(DataConfig):

    DELIMITER: str
    CONVERT_TO_LIST: list[str]
    SELECTED_COLS: list[str] 
    CONVERT_TO_INT: list[ConvertToIntConfig]

    def __init__(
        self, 
        FILE_NAME: str, 
        SELECTED_COLS: list[str], 
        ROW_LIMIT: int,
        CONVERT_TO_INT: list[ConvertToIntConfig],
        DELIMITER: str,
        CONVERT_TO_LIST: list[str],
    ) -> None:
        self.DELIMITER = DELIMITER
        self.CONVERT_TO_LIST = CONVERT_TO_LIST
        self.SELECTED_COLS = SELECTED_COLS
        self.CONVERT_TO_INT = CONVERT_TO_INT
        super().__init__(
            FILE_NAME=FILE_NAME,
            ROW_LIMIT=ROW_LIMIT,
        )


    def __str__(self) -> str:
        repr = {
            'file_name': self.FILE_NAME,
            'selected_cols': self.SELECTED_COLS,
            'limit': self.ROW_LIMIT,
            'convert_to_int': self.CONVERT_TO_INT,
            'convert_to_list': self.CONVERT_TO_LIST,
            'delimiter': self.DELIMITER,
        }
        return str(repr)