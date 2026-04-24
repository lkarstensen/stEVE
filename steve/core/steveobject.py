"""Base class for all steve components."""
from abc import ABC
import inspect
from typing import Any, TypeVar

from steve.core.confighandler import ConfigHandler
from steve.util.logging import get_logger

_logger = get_logger(__name__)

T = TypeVar("T", bound="StEveObject")


class StEveObject(ABC):
    """Base class for all steve components.

    Provides a YAML-based serialization contract: every ``__init__``
    parameter must be stored as a public attribute with the **same name**::

        def __init__(self, friction: float):
            self.friction = friction   # same name → auto-serialized

    Private attributes (``self._x``) and attributes computed from parameters
    are never serialized automatically. Override ``to_config`` and optionally
    ``from_config`` when the naming convention cannot be met.
    """

    def __repr__(self) -> str:
        """Return the fully-qualified class name as the string representation."""
        return f"{self.__module__}.{self.__class__.__name__}"

    # ------------------------------------------------------------------
    # Serialization hooks — override these to customise serialization
    # ------------------------------------------------------------------

    def to_config(self) -> dict:
        """Return the constructor kwargs needed to reconstruct this object.

        The default implementation reads every ``__init__`` parameter from
        the matching public attribute. Override when parameter names differ
        from attribute names, or when values require a transformation before
        saving.

        Returns:
            Mapping of constructor parameter names to their current values.
        """
        sig = inspect.signature(self.__init__)
        params = [
            name
            for name, param in sig.parameters.items()
            if param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,  # *args
                inspect.Parameter.VAR_KEYWORD,  # **kwargs
            )
        ]
        return {p: getattr(self, p) for p in params}

    @classmethod
    def from_config(cls: type[T], **kwargs: Any) -> T:
        """Reconstruct this object from config kwargs.

        The default calls ``cls(**kwargs)``. Override when reconstruction
        requires different logic than the ``__init__`` signature suggests.

        Args:
            **kwargs: Constructor arguments as returned by ``to_config``.

        Returns:
            A new instance of this class.
        """
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Public serialization API — thin delegates to ConfigHandler
    # ------------------------------------------------------------------

    def save_config(self, file_path: str) -> None:
        """Save this object's configuration to a YAML file.

        Args:
            file_path: Destination path. A ``.yml`` extension is appended
                if not already present.

        Raises:
            ValueError: If ``file_path`` is empty.
        """
        if not file_path:
            raise ValueError(
                f"{self.__class__.__name__}.save_config: "
                "file_path must be a non-empty string"
            )
        _logger.debug("Saving config %r → %s", self, file_path)
        ConfigHandler().save(self, file_path)

    def get_config_dict(self) -> dict:
        """Return this object's configuration as a nested dictionary.

        Returns:
            Nested dict suitable for serialization or inspection.
        """
        return ConfigHandler().to_dict(self)

    @classmethod
    def from_config_file(
        cls: type[T],
        config_file: str,
        exchange: dict[type, "StEveObject"] | None = None,
    ) -> T:
        """Load and reconstruct an object from a YAML config file.

        Args:
            config_file: Path to the YAML file produced by ``save_config``.
            exchange: Optional ``{OldType: replacement}`` mapping passed to
                ``reconstruct``. See ``ConfigHandler.reconstruct`` for details.

        Returns:
            The reconstructed object.

        Raises:
            ValueError: If the file's root class is not a subclass of ``cls``.
        """
        if not config_file:
            raise ValueError(
                f"{cls.__name__}.from_config_file: "
                "config_file must be a non-empty string"
            )
        _logger.debug("Loading %s from %s", cls.__name__, config_file)
        return ConfigHandler().load(
            config_file, expected_type=cls, exchange=exchange
        )

    @classmethod
    def from_config_dict(
        cls: type[T],
        config_dict: dict,
        exchange: dict[type, "StEveObject"] | None = None,
    ) -> T:
        """Reconstruct an object from a config dictionary.

        Args:
            config_dict: Dictionary produced by ``get_config_dict``.
            exchange: Optional ``{OldType: replacement}`` mapping. Wherever
                the config references an object of ``OldType``, the
                replacement is used and now-unused sub-objects are pruned.

        Returns:
            The reconstructed object.

        Raises:
            ValueError: If the config's root class is not a subclass of ``cls``.
        """
        _logger.debug("Reconstructing %s from config dict", cls.__name__)
        return ConfigHandler().reconstruct(
            config_dict, expected_type=cls, exchange=exchange
        )
