"""Tests for steve.core.confighandler.ConfigHandler."""
from enum import Enum
from typing import Any

import numpy as np
import pytest
from hypothesis import given, strategies as st

from steve.core.confighandler import ConfigHandler, _get_class
from steve.core.steveobject import StEveObject
from tests.unit._helpers import Container, Leaf, SharedPair


# ---------------------------------------------------------------------------
# Test-local StEveObject subclasses (not round-tripped, so don't need to be
# importable by _get_class — only used with to_dict)
# ---------------------------------------------------------------------------

class _Color(Enum):
    RED = "red"
    BLUE = "blue"


class _ArrayLeaf(StEveObject):
    def __init__(self, arr: Any) -> None:
        self.arr = arr


class _EnumLeaf(StEveObject):
    def __init__(self, color: _Color) -> None:
        self.color = color


class _TupleLeaf(StEveObject):
    def __init__(self, data: Any) -> None:
        self.data = data


class _Unserializable:
    pass


class _BadLeaf(StEveObject):
    def __init__(self, thing: Any) -> None:
        self.thing = thing


class _ListLeaf(StEveObject):
    def __init__(self, items: Any) -> None:
        self.items = items


# ---------------------------------------------------------------------------
# _get_class (module-level function)
# ---------------------------------------------------------------------------

class TestGetClass:
    """Tests for the module-level _get_class helper."""

    def test_resolves_known_class(self):
        cls = _get_class("tests.unit._helpers.Leaf")
        assert cls is Leaf

    def test_invalid_module_raises(self):
        with pytest.raises((ModuleNotFoundError, AttributeError)):
            _get_class("nonexistent.module.Foo")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _get_class("")

    def test_no_dot_raises(self):
        with pytest.raises(ValueError):
            _get_class("NoDotPath")

    def test_missing_attribute_raises(self):
        with pytest.raises(AttributeError):
            _get_class("builtins.NonExistentClass")


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    """Tests for ConfigHandler.to_dict."""

    def test_has_class_and_id(self):
        result = ConfigHandler().to_dict(Leaf(value=1.0))
        assert "_class" in result
        assert "_id" in result

    def test_class_is_fully_qualified(self):
        result = ConfigHandler().to_dict(Leaf(value=1.0))
        assert result["_class"] == "tests.unit._helpers.Leaf"

    def test_params_present(self):
        result = ConfigHandler().to_dict(Leaf(value=3.14, name="x"))
        assert result["value"] == 3.14
        assert result["name"] == "x"

    def test_nested_child_has_class_and_id(self):
        result = ConfigHandler().to_dict(Container(child=Leaf(value=1.0), tag="t"))
        assert "_class" in result["child"]
        assert "_id" in result["child"]

    def test_shared_subobject_second_reference_is_stub(self):
        leaf = Leaf(value=1.0)
        pair = SharedPair(first=leaf, second=leaf)
        result = ConfigHandler().to_dict(pair)
        first, second = result["first"], result["second"]
        assert len(first) > 2   # full entry has params
        assert set(second) == {"_class", "_id"}  # stub only

    def test_registry_cleared_between_calls(self):
        obj = Leaf(value=1.0)
        ch = ConfigHandler()
        r1 = ch.to_dict(obj)
        r2 = ch.to_dict(obj)
        assert r1["value"] == r2["value"]

    def test_numpy_array_serialized_to_list(self):
        result = ConfigHandler().to_dict(_ArrayLeaf(arr=np.array([1.0, 2.0, 3.0])))
        assert result["arr"] == [1.0, 2.0, 3.0]

    def test_numpy_integer_serialized_to_int(self):
        result = ConfigHandler().to_dict(_ArrayLeaf(arr=np.int64(42)))
        assert isinstance(result["arr"], int)
        assert result["arr"] == 42

    def test_enum_serialized_to_value(self):
        result = ConfigHandler().to_dict(_EnumLeaf(color=_Color.RED))
        assert result["color"] == "red"

    def test_tuple_serialized_to_tuple(self):
        result = ConfigHandler().to_dict(_TupleLeaf(data=(1, 2, 3)))
        assert result["data"] == (1, 2, 3)
        assert isinstance(result["data"], tuple)

    def test_unknown_object_raises(self):
        with pytest.raises(NotImplementedError):
            ConfigHandler().to_dict(_BadLeaf(thing=_Unserializable()))

    def test_list_param_serialized_to_list(self):
        result = ConfigHandler().to_dict(_ListLeaf(items=[1, "a", 2.5]))
        assert result["items"] == [1, "a", 2.5]
        assert isinstance(result["items"], list)

    def test_none_obj_raises(self):
        with pytest.raises(ValueError, match="obj must not be None"):
            ConfigHandler().to_dict(None)


# ---------------------------------------------------------------------------
# reconstruct
# ---------------------------------------------------------------------------

class TestReconstruct:
    """Tests for ConfigHandler.reconstruct."""

    def test_simple_roundtrip(self):
        obj = Leaf(value=2.5, name="foo")
        ch = ConfigHandler()
        restored = ch.reconstruct(ch.to_dict(obj))
        assert restored.value == obj.value
        assert restored.name == obj.name

    def test_nested_roundtrip(self):
        container = Container(child=Leaf(value=1.0, name="inner"), tag="outer")
        ch = ConfigHandler()
        restored = ch.reconstruct(ch.to_dict(container))
        assert restored.tag == "outer"
        assert restored.child.value == 1.0
        assert restored.child.name == "inner"

    def test_shared_subobject_is_same_instance(self):
        leaf = Leaf(value=1.0)
        pair = SharedPair(first=leaf, second=leaf)
        ch = ConfigHandler()
        restored = ch.reconstruct(ch.to_dict(pair))
        assert restored.first is restored.second

    def test_expected_type_correct_passes(self):
        ch = ConfigHandler()
        restored = ch.reconstruct(ch.to_dict(Leaf(value=1.0)), expected_type=Leaf)
        assert isinstance(restored, Leaf)

    def test_expected_type_parent_class_passes(self):
        ch = ConfigHandler()
        restored = ch.reconstruct(
            ch.to_dict(Leaf(value=1.0)), expected_type=StEveObject
        )
        assert isinstance(restored, StEveObject)

    def test_expected_type_mismatch_raises(self):
        ch = ConfigHandler()
        config = ch.to_dict(Leaf(value=1.0))
        with pytest.raises(ValueError, match="not a subclass"):
            ch.reconstruct(config, expected_type=Container)

    def test_expected_type_unknown_class_raises(self):
        bad_config = {"_class": "nonexistent.module.Foo", "_id": 1}
        with pytest.raises(ValueError, match="Cannot load class"):
            ConfigHandler().reconstruct(bad_config, expected_type=Leaf)

    def test_missing_class_key_raises(self):
        with pytest.raises(ValueError, match="_class"):
            ConfigHandler().reconstruct({"_id": 1})

    def test_missing_id_key_raises(self):
        with pytest.raises(ValueError, match="_id"):
            ConfigHandler().reconstruct({"_class": "foo.Bar"})

    def test_exchange_replaces_child(self):
        container = Container(child=Leaf(value=1.0), tag="t")
        replacement = Leaf(value=99.0, name="replaced")
        ch = ConfigHandler()
        restored = ch.reconstruct(ch.to_dict(container), exchange={Leaf: replacement})
        assert restored.child is replacement


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """Tests for ConfigHandler.save and ConfigHandler.load."""

    def test_save_appends_yml_extension(self, tmp_path):
        path = str(tmp_path / "config")
        ConfigHandler().save(Leaf(value=1.0), path)
        assert (tmp_path / "config.yml").exists()

    def test_save_does_not_double_append_extension(self, tmp_path):
        path = str(tmp_path / "config.yml")
        ConfigHandler().save(Leaf(value=1.0), path)
        assert (tmp_path / "config.yml").exists()
        assert not (tmp_path / "config.yml.yml").exists()

    def test_save_load_roundtrip(self, tmp_path):
        obj = Leaf(value=3.14, name="saved")
        ch = ConfigHandler()
        path = str(tmp_path / "config")
        ch.save(obj, path)
        restored = ch.load(path + ".yml")
        assert restored.value == obj.value
        assert restored.name == obj.name

    def test_load_with_correct_expected_type(self, tmp_path):
        ch = ConfigHandler()
        path = str(tmp_path / "config")
        ch.save(Leaf(value=1.0), path)
        restored = ch.load(path + ".yml", expected_type=Leaf)
        assert isinstance(restored, Leaf)

    def test_load_with_wrong_expected_type_raises(self, tmp_path):
        ch = ConfigHandler()
        path = str(tmp_path / "config")
        ch.save(Leaf(value=1.0), path)
        with pytest.raises(ValueError):
            ch.load(path + ".yml", expected_type=Container)

    def test_save_none_obj_raises(self):
        with pytest.raises(ValueError, match="obj must not be None"):
            ConfigHandler().save(None, "path")

    def test_save_empty_path_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ConfigHandler().save(Leaf(value=1.0), "")

    def test_load_empty_path_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ConfigHandler().load("")


# ---------------------------------------------------------------------------
# collect_object_ids
# ---------------------------------------------------------------------------

class TestCollectObjectIds:
    """Tests for ConfigHandler.collect_object_ids."""

    def test_single_object_returns_one_entry(self):
        ch = ConfigHandler()
        registry = ch.collect_object_ids(ch.to_dict(Leaf(value=1.0)))
        assert len(registry) == 1

    def test_nested_objects_returns_all_entries(self):
        ch = ConfigHandler()
        config = ch.to_dict(Container(child=Leaf(value=1.0)))
        registry = ch.collect_object_ids(config)
        assert len(registry) == 2

    def test_stub_resolved_from_full_config(self):
        leaf = Leaf(value=1.0)
        pair = SharedPair(first=leaf, second=leaf)
        ch = ConfigHandler()
        full_config = ch.to_dict(pair)
        second_stub = full_config["second"]
        leaf_id = second_stub["_id"]

        registry = ch.collect_object_ids(second_stub, full_config_dict=full_config)

        assert leaf_id in registry
        assert "value" in registry[leaf_id]  # resolved to full entry, not stub

    def test_missing_id_key_raises(self):
        with pytest.raises(ValueError, match="_id"):
            ConfigHandler().collect_object_ids({"_class": "foo.Bar"})

    def test_full_config_dict_missing_id_raises(self):
        ch = ConfigHandler()
        config = ch.to_dict(Leaf(value=1.0))
        with pytest.raises(ValueError, match="full_config_dict"):
            ch.collect_object_ids(config, full_config_dict={"_class": "foo.Bar"})


# ---------------------------------------------------------------------------
# Hypothesis round-trip
# ---------------------------------------------------------------------------

@given(
    value=st.floats(allow_nan=False, allow_infinity=False),
    name=st.text(min_size=1, max_size=64),
)
def test_leaf_roundtrip_hypothesis(value: float, name: str) -> None:
    obj = Leaf(value=value, name=name)
    ch = ConfigHandler()
    restored = ch.reconstruct(ch.to_dict(obj))
    assert restored.value == obj.value
    assert restored.name == name
