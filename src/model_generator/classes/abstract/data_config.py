from .config import Config

class DataConfig(Config):

    FILE_NAME: str
    ROW_LIMIT: int

    def __init__(
        self, 
        FILE_NAME: str,
        ROW_LIMIT: int, 
    ) -> None:
        self.FILE_NAME = FILE_NAME
        self.ROW_LIMIT = ROW_LIMIT
        super().__init__()


    def __str__(self) -> str:
        repr = {
            'file_name': self.FILE_NAME,
            'limit': self.ROW_LIMIT,
        }
        return str(repr)