from abc import ABC


class EveObject(ABC):
    def __repr__(self):
        return f"{self.__module__}.{self.__class__.__name__}"
