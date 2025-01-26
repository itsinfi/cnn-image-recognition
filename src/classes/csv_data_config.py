from .convert_to_int_config import ConvertToIntConfig
from .abstract.data_config import DataConfig

class CsvDataConfig(DataConfig):

    DELIMITER: str
    CONVERT_TO_LIST: list[str]

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
        super().__init__(
            FILE_NAME=FILE_NAME,
            SELECTED_COLS=SELECTED_COLS,
            ROW_LIMIT=ROW_LIMIT,
            CONVERT_TO_INT=CONVERT_TO_INT,
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