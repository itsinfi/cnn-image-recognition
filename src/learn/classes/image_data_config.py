from .abstract.data_config import DataConfig

class ImageDataConfig(DataConfig):

    DELIMITER: str
    FILTER_LIST: list[str]
    IMAGE_COMPR_SIZE: tuple[int, int]

    def __init__(
        self, 
        FILE_NAME: str,
        ROW_LIMIT: int,
        DELIMITER: str,
        FILTER_LIST: list[str],
        IMAGE_COMPR_SIZE: tuple[int, int],
    ) -> None:
        self.DELIMITER = DELIMITER
        self.FILTER_LIST = FILTER_LIST
        self.IMAGE_COMPR_SIZE = IMAGE_COMPR_SIZE
        super().__init__(
            FILE_NAME=FILE_NAME,
            ROW_LIMIT=ROW_LIMIT,
        )


    def __str__(self) -> str:
        repr = {
            'file_name': self.FILE_NAME,
            'limit': self.ROW_LIMIT,
            'delimiter': self.DELIMITER,
            'filter_list': self.FILTER_LIST,
            'image_compression_size': self.IMAGE_COMPR_SIZE,
        }
        return str(repr)