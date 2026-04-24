"""Serialization and deserialization of StEveObject graphs to/from YAML."""

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from enum import Enum
from importlib import import_module
from typing import Any

import numpy as np
import yaml

from steve.util.logging import get_logger

_logger = get_logger(__name__)

try:
    from yaml import CLoader as _YamlLoader
except ImportError:
    from yaml import Loader as _YamlLoader  # type: ignore[assignment]


def _get_class(class_str: str) -> type:
    """Import and return the class named by a dotted module path.

    Args:
        class_str: Dotted ``module.ClassName`` string.

    Returns:
        The imported class object.

    Raises:
        ValueError: If ``class_str`` is empty or contains no dot.
    """
    if not class_str:
        raise ValueError(
            "ConfigHandler._get_class: class_str must be a"
            " non-empty dotted module path, got empty string."
        )
    if "." not in class_str:
        raise ValueError(
            f"ConfigHandler._get_class: class_str must be a"
            f" dotted 'module.ClassName' string, got {class_str!r}."
        )
    module_path, class_name = class_str.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


class ConfigHandler:
    """Serializes and deserializes StEveObject graphs to/from YAML config dicts.

    Each StEveObject exposes its constructor kwargs via ``to_config()`` and is
    reconstructed via ``from_config(**kwargs)``. Shared sub-objects are
    serialized once (keyed by Python ``id``) and resolved correctly on load.
    """

    # — public API —

    def save(self, obj: Any, file_path: str) -> None:
        """Serialize ``obj`` and write to a YAML file.

        Args:
            obj: The StEveObject to serialize.
            file_path: Destination path. A ``.yml`` extension is appended
                if not already present.

        Raises:
            ValueError: If ``obj`` is ``None`` or ``file_path`` is empty.
        """
        if obj is None:
            raise ValueError("ConfigHandler.save: obj must not be None.")
        if not file_path:
            raise ValueError(
                "ConfigHandler.save: file_path must be a non-empty string."
            )
        config_dict = self.to_dict(obj)
        final_path = self._write_yaml(config_dict, file_path)
        _logger.info("Config saved to %r", final_path)

    def to_dict(self, obj: Any) -> dict:
        """Serialize ``obj`` to a nested config dictionary.

        Args:
            obj: The StEveObject to serialize.

        Returns:
            Nested dict with ``_class`` and ``_id`` metadata alongside the
            constructor kwargs returned by ``obj.to_config()``.

        Raises:
            ValueError: If ``obj`` is ``None``.
        """
        if obj is None:
            raise ValueError("ConfigHandler.to_dict: obj must not be None.")
        _logger.debug("Serializing %s", obj.__class__.__name__)
        registry: dict[int, Any] = {}
        result = self._serialize_steve_obj(obj, registry)
        _logger.debug("Serialization complete: %s", obj.__class__.__name__)
        return result

    def load(
        self,
        file_path: str,
        expected_type: type | None = None,
        exchange: Mapping[type, Any] | None = None,
    ) -> Any:
        """Read a YAML file and reconstruct the stored object.

        Args:
            file_path: Path to a YAML file produced by ``save``.
            expected_type: When given, raises ``ValueError`` if the root
                object is not a subclass of this type.
            exchange: Optional ``{OldType: replacement}`` mapping forwarded
                to ``reconstruct``.

        Returns:
            The reconstructed StEveObject.

        Raises:
            ValueError: If ``file_path`` is empty, or if ``expected_type``
                is given and the root class does not match.
        """
        if not file_path:
            raise ValueError(
                "ConfigHandler.load: file_path must be a non-empty string."
            )
        _logger.info("Loading config from %r", file_path)
        config_dict = self._read_yaml(file_path)
        return self.reconstruct(
            config_dict, expected_type=expected_type, exchange=exchange
        )

    def reconstruct(
        self,
        config_dict: MutableMapping[str, Any],
        expected_type: type | None = None,
        exchange: Mapping[type, Any] | None = None,
    ) -> Any:
        """Reconstruct an object from a config dictionary.

        Args:
            config_dict: Dictionary produced by ``to_dict``.
            expected_type: When given, raises ``ValueError`` if the root
                object is not a subclass of this type.
            exchange: ``{OldType: replacement_instance}`` mapping. Wherever
                the config references an object of ``OldType``, the
                replacement is used and now-unused sub-objects are pruned.

        Returns:
            The reconstructed StEveObject.

        Raises:
            ValueError: If ``config_dict`` is missing ``_class`` or ``_id``
                keys, or if ``expected_type`` is given and the root class
                does not match.
        """
        if "_class" not in config_dict or "_id" not in config_dict:
            raise ValueError(
                "ConfigHandler.reconstruct: config_dict must contain"
                " '_class' and '_id' keys, got keys "
                f"{list(config_dict.keys())!r}."
            )
        config_dict = deepcopy(config_dict)
        _logger.debug("Reconstructing %r", config_dict.get("_class", "?"))

        if expected_type is not None:
            self._validate_type(config_dict, expected_type)

        object_registry: dict[int, Any] = {}
        if exchange:
            config_dict, object_registry = self._apply_exchange(
                config_dict, exchange
            )

        result = self._deserialize_steve_obj(config_dict, object_registry)
        _logger.debug("Reconstruct complete: %s", result.__class__.__name__)
        return result

    def collect_object_ids(
        self,
        config_dict: Mapping[str, Any],
        full_config_dict: Mapping[str, Any] | None = None,
    ) -> dict[int, dict]:
        """Return a flat ``{id: entry}`` registry for all reachable objects.

        Args:
            config_dict: The sub-tree to collect from.
            full_config_dict: The complete config dict. When supplied, stub
                entries (objects referenced by ``{_class, _id}`` only) are
                resolved against it to recover their full definitions.

        Returns:
            Mapping of object id to its config entry dict.

        Raises:
            ValueError: If ``config_dict`` does not contain an ``_id`` key,
                or if ``full_config_dict`` is provided and does not contain
                an ``_id`` key.
        """
        if "_id" not in config_dict:
            raise ValueError(
                "ConfigHandler.collect_object_ids: config_dict must"
                " contain an '_id' key, got keys "
                f"{list(config_dict.keys())!r}."
            )
        if full_config_dict is not None and "_id" not in full_config_dict:
            raise ValueError(
                "ConfigHandler.collect_object_ids: full_config_dict must"
                " contain an '_id' key, got keys "
                f"{list(full_config_dict.keys())!r}."
            )
        full_registry: dict[int, dict] = {}
        if full_config_dict is not None:
            self._build_id_registry(full_config_dict, full_registry, {})

        result: dict[int, dict] = {}
        self._build_id_registry(config_dict, result, full_registry)
        _logger.debug("collect_object_ids: found %d objects", len(result))
        return result

    # — YAML helpers —

    def _write_yaml(
        self, config_dict: Mapping[str, Any], file_path: str
    ) -> str:
        """Write ``config_dict`` to ``file_path``, appending ``.yml`` if absent.

        Returns:
            The final path written (may differ from ``file_path`` by ``.yml``).
        """
        if not file_path.endswith(".yml"):
            file_path += ".yml"
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=4,
            )
        return file_path

    def _read_yaml(self, file_path: str) -> dict:
        """Read and parse a YAML file, returning the top-level dict."""
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=_YamlLoader)

    # — serialization —

    def _serialize_steve_obj(self, obj: Any, registry: dict[int, Any]) -> dict:
        """Serialize an StEveObject to a ``{_class, _id, **params}`` dict."""
        class_str = f"{obj.__module__}.{obj.__class__.__name__}"
        obj_id = id(obj)

        result: dict = {"_class": class_str, "_id": obj_id}

        if obj_id in registry:
            # Already serialized fully — emit a stub so the deserializer
            # can look it up in its registry.
            return result

        # Process children before marking as registered (circular references
        # are not supported).
        raw_config = obj.to_config()
        for param_name, value in raw_config.items():
            result[param_name] = self._serialize_value(value, registry)

        registry[obj_id] = obj
        return result

    def _serialize_value(  # pylint: disable=too-many-return-statements
        self, value: Any, registry: dict[int, Any]
    ) -> Any:
        """Recursively convert a value to YAML-safe native types."""
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return [self._serialize_value(v, registry) for v in value]
        if isinstance(value, tuple):
            return tuple(self._serialize_value(v, registry) for v in value)
        if isinstance(value, dict):
            return {k: self._serialize_value(v, registry) for k, v in value.items()}

        # Imported lazily: steve.core imports ConfigHandler, so a top-level
        # `from steve.core import StEveObject` here would create a cycle.
        steve_core = import_module("steve.core")
        if isinstance(value, steve_core.StEveObject):
            return self._serialize_steve_obj(value, registry)

        if hasattr(value, "__module__"):
            _logger.error(
                "Serialization of %s not implemented", value.__class__
            )
            raise NotImplementedError(
                f"Serialization of {value.__class__} is not implemented."
            )
        return value

    # — deserialization —

    def _deserialize_steve_obj(
        self, config_dict: Mapping[str, Any], registry: dict[int, Any]
    ) -> Any:
        """Reconstruct a StEveObject from a ``{_class, _id, **params}`` dict."""
        config_dict = dict(config_dict)  # shallow copy — we pop keys
        obj_id = config_dict.pop("_id")

        if obj_id in registry:
            return registry[obj_id]

        class_str = config_dict.pop("_class")
        cls = _get_class(class_str)

        kwargs = {
            k: self._deserialize_value(v, registry)
            for k, v in config_dict.items()
        }

        obj = cls.from_config(**kwargs)
        registry[obj_id] = obj
        return obj

    def _deserialize_value(self, value: Any, registry: dict[int, Any]) -> Any:
        """Recursively reconstruct a value from its serialized form."""
        if isinstance(value, dict):
            if "_class" in value and "_id" in value:
                return self._deserialize_steve_obj(value, registry)
            return {k: self._deserialize_value(v, registry) for k, v in value.items()}
        if isinstance(value, list):
            return [self._deserialize_value(v, registry) for v in value]
        if isinstance(value, tuple):
            return tuple(self._deserialize_value(v, registry) for v in value)
        return value

    # — exchange / dependency pruning —

    def _apply_exchange(  # pylint: disable=too-many-locals
        self,
        config_dict: MutableMapping[str, Any],
        exchange: Mapping[type, Any],
    ) -> tuple[MutableMapping[str, Any], dict[int, Any]]:
        """Replace objects in ``config_dict`` by type and prune unused deps.

        Args:
            config_dict: The serialized object graph to modify in place.
            exchange: ``{OldType: replacement_instance}`` mapping.

        Returns:
            Tuple of the pruned config dict and a pre-populated object
            registry mapping each replaced id to its replacement.
        """
        object_list: dict[int, Any] = {}
        self._build_id_registry(config_dict, object_list, {})

        # Clear the root Env's requires list so its direct children can be
        # pruned when replaced.
        steve = import_module("steve")
        root_cls = _get_class(config_dict["_class"])
        if issubclass(root_cls, steve.Env):
            object_list[config_dict["_id"]]["requires"] = []

        object_registry: dict[int, Any] = {}
        maybe_orphaned: list[int] = []

        for target_type, replacement in exchange.items():
            for obj_id, entry in list(object_list.items()):
                if not isinstance(entry, dict):
                    continue
                obj_cls = _get_class(entry["_class"])
                if issubclass(obj_cls, target_type):
                    maybe_orphaned.extend(entry.get("requires", []))
                    object_registry[obj_id] = replacement
                    object_list[obj_id] = replacement

        maybe_orphaned = list(set(maybe_orphaned))
        still_required = set()
        for obj_id in maybe_orphaned:
            for entry in object_list.values():
                if not isinstance(entry, dict):
                    continue
                if obj_id in entry.get("requires", []):
                    still_required.add(obj_id)

        pruned = []
        for obj_id in maybe_orphaned:
            if obj_id not in still_required:
                object_list.pop(obj_id, None)
                pruned.append(obj_id)
        _logger.debug(
            "Exchange applied: %d replaced, %d orphans pruned",
            len(object_registry),
            len(pruned),
        )

        to_remove = [
            key
            for key, val in config_dict.items()
            if isinstance(val, dict) and val.get("_id") not in object_list
        ]
        for key in to_remove:
            config_dict.pop(key)

        return config_dict, object_registry

    # — ID-registry builder —

    def _build_id_registry(
        self,
        config_dict: Mapping[str, Any],
        registry: MutableMapping[int, dict],
        full_registry: Mapping[int, dict],
    ) -> None:
        """Register all StEveObject entries reachable from ``config_dict``."""
        obj_id = config_dict["_id"]
        if obj_id in registry:
            return

        resolved: dict = dict(full_registry.get(obj_id, config_dict))

        entry: dict = deepcopy(resolved)
        entry["requires"] = []
        registry[obj_id] = entry

        for value in resolved.values():
            child_ids = self._collect_value_ids(value, registry, full_registry)
            entry["requires"].extend(child_ids)

    def _collect_value_ids(
        self,
        value: Any,
        registry: MutableMapping[int, dict],
        full_registry: Mapping[int, dict],
    ) -> list[int]:
        """Walk a serialized value, registering any nested StEveObject dicts."""
        ids: list[int] = []
        if isinstance(value, (list, tuple)):
            for v in value:
                ids.extend(self._collect_value_ids(v, registry, full_registry))
        elif isinstance(value, dict):
            if "_id" in value and "_class" in value:
                self._build_id_registry(value, registry, full_registry)
                ids.append(value["_id"])
            else:
                for v in value.values():
                    ids.extend(self._collect_value_ids(v, registry, full_registry))
        return ids

    # — utilities —

    def _validate_type(
        self, config_dict: Mapping[str, Any], expected_type: type
    ) -> None:
        """Check root class is a subclass of ``expected_type``; raise if not."""
        class_str = config_dict.get("_class", "")
        try:
            actual_cls = _get_class(class_str)
        except (AttributeError, ModuleNotFoundError, ValueError) as exc:
            raise ValueError(f"Cannot load class '{class_str}': {exc}") from exc
        if not issubclass(actual_cls, expected_type):
            raise ValueError(
                f"Config is for {class_str!r}, which is not a subclass of "
                f"{expected_type.__name__!r}."
            )
