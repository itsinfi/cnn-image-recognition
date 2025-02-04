from abc import ABC, abstractmethod

class Config(ABC):

    @abstractmethod
    def __str__(self):
        pass