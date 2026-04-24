"""Tests for steve.core.steveobject.StEveObject."""
import pytest
from hypothesis import given, strategies as st

from steve.core.steveobject import StEveObject
from tests.unit._helpers import Container, Leaf


# ---------------------------------------------------------------------------
# Local test-only subclasses
# ---------------------------------------------------------------------------

class _VarArgsObj(StEveObject):
    """StEveObject whose __init__ has *args and **kwargs — both must be excluded."""

    def __init__(self, value: int, *args, **kwargs) -> None:
        self.value = value

    @classmethod
    def from_config(cls, **kwargs):
        return cls(kwargs["value"])


class _DefaultsObj(StEveObject):
    """StEveObject with optional parameters to test default-value handling."""

    def __init__(self, required: str, optional: int = 0) -> None:
        self.required = required
        self.optional = optional


class _CustomConfigObj(StEveObject):
    """StEveObject that overrides to_config and from_config."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def to_config(self) -> dict:
        return {"x": self.x, "y": self.y, "sum": self.x + self.y}

    @classmethod
    def from_config(cls, **kwargs):
        return cls(x=kwargs["x"], y=kwargs["y"])


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    """Tests for StEveObject.__repr__."""

    def test_format_is_module_dot_classname(self):
        obj = Leaf(value=1.0)
        assert repr(obj) == "tests.unit._helpers.Leaf"

    def test_reflects_actual_class_not_base(self):
        obj = Container(child=Leaf(value=0.0))
        assert "Container" in repr(obj)


# ---------------------------------------------------------------------------
# to_config
# ---------------------------------------------------------------------------

class TestToConfig:
    """Tests for StEveObject.to_config."""

    def test_returns_init_params(self):
        obj = Leaf(value=3.14, name="x")
        config = obj.to_config()
        assert config == {"value": 3.14, "name": "x"}

    def test_reflects_current_attribute_values(self):
        obj = Leaf(value=1.0)
        obj.value = 99.0
        assert obj.to_config()["value"] == 99.0

    def test_excludes_var_positional_and_var_keyword(self):
        obj = _VarArgsObj(value=7)
        assert set(obj.to_config().keys()) == {"value"}

    def test_includes_params_with_defaults(self):
        obj = _DefaultsObj(required="hi")
        config = obj.to_config()
        assert "required" in config
        assert "optional" in config

    def test_default_value_preserved(self):
        obj = _DefaultsObj(required="hi")
        assert obj.to_config()["optional"] == 0

    def test_overridden_default_preserved(self):
        obj = _DefaultsObj(required="hi", optional=5)
        assert obj.to_config()["optional"] == 5

    def test_custom_to_config_used(self):
        obj = _CustomConfigObj(x=1.0, y=2.0)
        config = obj.to_config()
        assert config["sum"] == 3.0


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------

class TestFromConfig:
    """Tests for StEveObject.from_config."""

    def test_constructs_instance(self):
        obj = Leaf.from_config(value=1.0, name="a")
        assert isinstance(obj, Leaf)
        assert obj.value == 1.0
        assert obj.name == "a"

    def test_roundtrip_default_impl(self):
        original = Leaf(value=2.5, name="b")
        restored = Leaf.from_config(**original.to_config())
        assert restored.value == original.value
        assert restored.name == original.name

    def test_custom_from_config_used(self):
        obj = _CustomConfigObj.from_config(x=3.0, y=4.0, sum=7.0)
        assert obj.x == 3.0
        assert obj.y == 4.0


# ---------------------------------------------------------------------------
# get_config_dict / from_config_dict
# ---------------------------------------------------------------------------

class TestGetConfigDictAndFromConfigDict:
    """Tests for StEveObject.get_config_dict and StEveObject.from_config_dict."""

    def test_roundtrip(self):
        original = Leaf(value=1.5, name="cfg")
        restored = Leaf.from_config_dict(original.get_config_dict())
        assert restored.value == original.value
        assert restored.name == original.name

    def test_nested_roundtrip(self):
        original = Container(child=Leaf(value=7.0, name="inner"), tag="outer")
        restored = Container.from_config_dict(original.get_config_dict())
        assert restored.tag == "outer"
        assert restored.child.value == 7.0

    def test_wrong_expected_type_raises(self):
        config = Leaf(value=1.0).get_config_dict()
        with pytest.raises(ValueError):
            Container.from_config_dict(config)

    def test_get_config_dict_has_class_key(self):
        config = Leaf(value=1.0).get_config_dict()
        assert "_class" in config


# ---------------------------------------------------------------------------
# save_config / from_config_file
# ---------------------------------------------------------------------------

class TestSaveConfigAndFromConfigFile:
    """Tests for StEveObject.save_config and StEveObject.from_config_file."""

    def test_roundtrip(self, tmp_path):
        original = Leaf(value=9.9, name="file")
        path = str(tmp_path / "leaf")
        original.save_config(path)
        restored = Leaf.from_config_file(path + ".yml")
        assert restored.value == original.value
        assert restored.name == original.name

    def test_wrong_expected_type_raises(self, tmp_path):
        path = str(tmp_path / "leaf")
        Leaf(value=1.0).save_config(path)
        with pytest.raises(ValueError):
            Container.from_config_file(path + ".yml")

    def test_save_config_empty_path_raises(self):
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            Leaf(value=1.0).save_config("")

    def test_from_config_file_empty_path_raises(self):
        with pytest.raises(ValueError, match="config_file must be a non-empty string"):
            Leaf.from_config_file("")


# ---------------------------------------------------------------------------
# Hypothesis round-trip
# ---------------------------------------------------------------------------

@given(
    value=st.floats(allow_nan=False, allow_infinity=False),
    name=st.text(min_size=1, max_size=64),
)
def test_leaf_to_config_from_config_hypothesis(value: float, name: str) -> None:
    original = Leaf(value=value, name=name)
    restored = Leaf.from_config(**original.to_config())
    assert restored.value == original.value
    assert restored.name == original.name
