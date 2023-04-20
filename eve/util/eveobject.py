from abc import ABC
from importlib import import_module
from typing import Dict, Optional

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

    @classmethod
    def from_config_file(cls, config_file: str, to_exchange: Optional[Dict] = None):
        confighandler = ConfigHandler()
        config_dict = confighandler.load_config_dict(config_file)
        return cls.from_config_dict(config_dict, to_exchange)

    @classmethod
    def from_config_dict(cls, config_dict: Dict, to_exchange: Optional[Dict] = None):
        to_exchange = to_exchange or {}
        # check if correct class
        class_str = config_dict["_class"]
        module_path, class_name = class_str.rsplit(".", 1)
        module = import_module(module_path)
        new_obj_constructor = getattr(module, class_name)

        if cls != new_obj_constructor:
            raise ValueError("Config File from wrong class")
        confighandler = ConfigHandler()

        # get list of objects
        object_list = confighandler.config_dict_to_list_of_objects(config_dict)
        eve = import_module("eve")
        if issubclass(cls, eve.Env):
            object_list[config_dict["_id"]]["requires"] = []
        # exchange objects
        object_registry = {}
        maybe_no_longer_required = []
        to_pop = []

        # create object registry of exchanged objects
        for obj_class_to_exchange, obj in to_exchange.items():
            for obj_id, obj_dict in object_list.items():
                if not isinstance(obj_dict, dict):
                    continue
                class_str = obj_dict["_class"]
                module_path, class_name = class_str.rsplit(".", 1)
                module = import_module(module_path)
                current_obj = getattr(module, class_name)
                if issubclass(current_obj, obj_class_to_exchange):
                    maybe_no_longer_required += object_list[obj_id]["requires"]
                    to_pop.append(obj_id)
                    object_registry[obj_id] = obj
                    object_list[obj_id] = obj

        # find objects which dependencies were removed due to the exchange
        maybe_no_longer_required = list(set(maybe_no_longer_required))
        still_required = []
        for obj_id in maybe_no_longer_required:
            if obj_id in still_required:
                continue
            for obj_dict in object_list.values():
                if isinstance(obj_dict, dict) and obj_id in obj_dict["requires"]:
                    still_required.append(obj_id)
                    continue

        # remove objects without depenencies from registry
        for obj_id in maybe_no_longer_required:
            if not obj_id in still_required:
                object_list.pop(obj_id)

        # reduce config dict with only the ids left in reduced_object_registry
        to_remove = []
        for entry, obj_dict in config_dict.items():
            if isinstance(obj_dict, dict) and obj_dict["_id"] not in object_list.keys():
                to_remove.append(entry)

        for entry in to_remove:
            config_dict.pop(entry)

        return confighandler.config_dict_to_object(config_dict, object_registry)
