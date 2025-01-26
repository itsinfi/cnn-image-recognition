from .abstract.config import Config

class JoinConfig(Config):

    JOINS: str
    ON: str

    def __init__(self, JOINS: str, ON: str):
        self.JOINS = JOINS
        self.ON = ON
        super().__init__()
    
    def __str__(self):
        repr = {
            'joins': self.FILE_NAME,
            'on': self.ON,
        }
        return str(repr)