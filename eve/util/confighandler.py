from copy import deepcopy
import importlib
from typing import Any, Dict, Optional
from enum import Enum
from importlib import import_module

import inspect
import numpy as np
import yaml


class ConfigHandler:
    def __init__(self):
        self.object_registry = {}
        self._eve = import_module("eve")

    def save_config(self, eve_object: Any, file: str) -> None:
        obj_dict = self.object_to_config_dict(eve_object)
        self.save_config_dict(obj_dict, file)

    def object_to_config_dict(self, eve_object: Any) -> dict:
        self.object_registry = {}
        config_dict = self._eve_obj_to_dict(eve_object)
        self.object_registry = {}
        return config_dict

    def save_config_dict(self, config_dict: dict, file: str) -> None:
        if not file.endswith(".yml"):
            file += ".yml"
        with open(file, "w", encoding="utf-8") as dumpfile:
            yaml.dump(
                config_dict,
                dumpfile,
                default_flow_style=False,
                sort_keys=False,
                indent=4,
            )

    def load_config(self, file: str) -> Any:
        obj_dict = self.load_config_dict(file)
        obj = self.config_dict_to_object(obj_dict)
        return obj

    def load_config_dict(self, file: str) -> dict:
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        with open(file, "r", encoding="utf-8") as config:
            config_dict = yaml.load(config, Loader=Loader)
        return config_dict

    def config_dict_to_object(
        self,
        config_dict: dict,
        object_registry: Dict[int, object] = None,
    ) -> Any:
        self.object_registry = object_registry or {}
        obj = self._eve_config_dict_to_obj_recursive(config_dict)
        self.object_registry = {}
        return obj

    def config_dict_to_list_of_objects(
        self, config_dict: dict, full_config_dict: Optional[dict] = None
    ) -> Dict[int, str]:
        if full_config_dict is not None:
            full_config_registry, _ = self._config_dict_to_object_list_recursive(
                full_config_dict
            )
        else:
            full_config_registry = None
        config_list, _ = self._config_dict_to_object_list_recursive(
            config_dict, object_list=None, full_config_registry=full_config_registry
        )
        return config_list

    def _config_dict_to_object_list_recursive(
        self, config_dict: dict, object_list=None, full_config_registry: dict = None
    ) -> Dict[int, str]:
        object_list = object_list or {}
        full_config_registry = full_config_registry or {}
        obj_id = config_dict["_id"]
        if obj_id in object_list.keys():
            return object_list, obj_id

        if obj_id in full_config_registry.keys():
            config_dict = full_config_registry[obj_id]

        object_list[obj_id] = deepcopy(config_dict)
        object_list[obj_id]["requires"] = []
        for value in config_dict.values():
            (
                object_list,
                new_obj_id,
            ) = self._config_value_to_object_list_recursive(
                value, object_list, full_config_registry
            )
            if new_obj_id is not None:
                object_list[obj_id]["requires"].append(new_obj_id)
        return object_list, obj_id

    def _config_value_to_object_list_recursive(
        self, config_value, object_list: dict, full_config_registry
    ):
        obj_id = None
        if isinstance(config_value, (list, tuple)):
            for value in config_value:
                (
                    object_list,
                    obj_id,
                ) = self._config_value_to_object_list_recursive(
                    value, object_list, full_config_registry
                )
        elif isinstance(config_value, dict):
            if "_id" in config_value.keys() and "_class" in config_value.keys():
                (
                    object_list,
                    obj_id,
                ) = self._config_dict_to_object_list_recursive(
                    config_value, object_list, full_config_registry
                )
            else:
                for value in config_value.values():
                    (
                        object_list,
                        obj_id,
                    ) = self._config_value_to_object_list_recursive(
                        value, object_list, full_config_registry
                    )
        return object_list, obj_id

    def _eve_obj_to_dict(self, eve_object) -> dict:
        attributes_dict = {}
        attributes_dict[
            "_class"
        ] = f"{eve_object.__module__}.{eve_object.__class__.__name__}"
        object_id = id(eve_object)
        attributes_dict["_id"] = object_id
        if object_id in self.object_registry:
            return attributes_dict

        if isinstance(eve_object, self._eve.Env):
            init_attributes = self._get_init_attributes(self._eve.Env.__init__)
            if "self" in init_attributes:
                init_attributes.remove("self")
        else:
            init_attributes = self._get_init_attributes(eve_object.__init__)

        if "args" in init_attributes:
            init_attributes.remove("args")
        if "kwargs" in init_attributes:
            init_attributes.remove("kwargs")
        if "kwds" in init_attributes:
            init_attributes.remove("kwds")

        for attribute in init_attributes:
            nested_object = getattr(eve_object, attribute)
            attributes_dict[attribute] = self._obj_to_native_datatypes(nested_object)

        self.object_registry[object_id] = eve_object
        return attributes_dict

    def _obj_to_native_datatypes(self, obj) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, list):
            return [self._obj_to_native_datatypes(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._obj_to_native_datatypes(v) for v in obj)

        if isinstance(obj, dict):
            return {k: self._obj_to_native_datatypes(v) for k, v in obj.items()}

        eve = importlib.import_module("eve")
        if isinstance(obj, eve.util.EveObject):
            return self._eve_obj_to_dict(obj)

        if hasattr(obj, "__module__"):
            raise NotImplementedError(
                f"Handling this class {obj.__class__} in not implemented "
            )

        return obj

    @staticmethod
    def _get_init_attributes(init_function):
        init_attributes = []
        kwargs = inspect.signature(init_function)
        for param in kwargs.parameters.values():
            init_attributes.append(param.name)

        return init_attributes

    def _eve_config_dict_to_obj_recursive(self, obj_config_dict: dict):
        obj_id = obj_config_dict.pop("_id")
        if obj_id in self.object_registry.keys():
            return self.object_registry[obj_id]

        class_str: str = obj_config_dict.pop("_class")
        obj_kwds = {}
        for attribute_name, value in obj_config_dict.items():
            obj_kwds[attribute_name] = self._config_dict_value_converter(value)

        constructor = self._get_class_constructor(class_str)
        obj = constructor(**obj_kwds)
        self.object_registry[obj_id] = obj
        return obj

    def _config_dict_value_converter(self, value):
        if isinstance(value, dict):
            if "_class" in value.keys() and "_id" in value.keys():
                return self._eve_config_dict_to_obj_recursive(value)
            return {k: self._config_dict_value_converter(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._config_dict_value_converter(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._config_dict_value_converter(v) for v in value)
        return value

    def _get_class_constructor(self, class_str: str):
        module_path, class_name = class_str.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
