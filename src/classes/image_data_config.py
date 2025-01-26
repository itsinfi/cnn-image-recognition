from .convert_to_int_config import ConvertToIntConfig
from .abstract.data_config import DataConfig

class ImageDataConfig(DataConfig):

    DELIMITER: str

    def __init__(
        self, 
        FILE_NAME: str,
        ROW_LIMIT: int,
        DELIMITER: str,
    ) -> None:
        self.DELIMITER = DELIMITER
        super().__init__(
            FILE_NAME=FILE_NAME,
            ROW_LIMIT=ROW_LIMIT,
        )


    def __str__(self) -> str:
        repr = {
            'file_name': self.FILE_NAME,
            'limit': self.ROW_LIMIT,
            'delimiter': self.DELIMITER,
        }
        return str(repr)