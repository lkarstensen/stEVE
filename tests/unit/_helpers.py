"""Minimal StEveObject subclasses used as test fixtures.

These must live in an importable module so that ConfigHandler._get_class
can reconstruct them by dotted path during round-trip tests.
"""
from steve.core import StEveObject


class Leaf(StEveObject):
    """Flat StEveObject with a float value and a string name."""

    def __init__(self, value: float, name: str = "leaf") -> None:
        self.value = value
        self.name = name


class Container(StEveObject):
    """StEveObject that holds a single child StEveObject."""

    def __init__(self, child: Leaf, tag: str = "") -> None:
        self.child = child
        self.tag = tag


class SharedPair(StEveObject):
    """Holds two references that may point to the same object."""

    def __init__(self, first: Leaf, second: Leaf) -> None:
        self.first = first
        self.second = second
