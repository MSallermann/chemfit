"""Helpers for defining identity-based sentinel values."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from typing_extensions import Self


class CopyStableSentinel:
    """Define one copy-stable sentinel instance per concrete subclass."""

    __slots__ = ()

    _instances: ClassVar[dict[type[CopyStableSentinel], CopyStableSentinel]] = {}

    def __new__(cls) -> Self:
        instance = cls._instances.get(cls)
        if instance is None:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cast("Self", instance)

    def __copy__(self) -> Self:
        """Preserve identity under shallow copying."""

        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Preserve identity under recursive copying."""

        memo[id(self)] = self
        return self
