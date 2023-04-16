from abc import ABC

from .confighandler import ConfigHandler


class EveObject(ABC):
    def __repr__(self):
        return f"{self.__module__}.{self.__class__.__name__}"

    def save_config(self, file_path: str):
        confighandler = ConfigHandler()
        confighandler.save_config(self, file_path)

    def get_config_dict(self):
        confighandler = ConfigHandler()
        return confighandler.object_to_config_dict(self)
