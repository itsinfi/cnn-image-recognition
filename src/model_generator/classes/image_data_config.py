from .convert_to_int_config import ConvertToIntConfig
from .abstract.data_config import DataConfig

class ImageDataConfig(DataConfig):

    DELIMITER: str

    def __init__(
        self, 
        FILE_NAME: str,
        ROW_LIMIT: int,
        DELIMITER: str,
        FILTER_LIST: list[str],
    ) -> None:
        self.DELIMITER = DELIMITER
        self.FILTER_LIST = FILTER_LIST
        super().__init__(
            FILE_NAME=FILE_NAME,
            ROW_LIMIT=ROW_LIMIT,
        )


    def __str__(self) -> str:
        repr = {
            'file_name': self.FILE_NAME,
            'limit': self.ROW_LIMIT,
            'delimiter': self.DELIMITER,
            'filters': self.FILTER_LIST,
        }
        return str(repr)