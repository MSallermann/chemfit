"""Versioned, JSON-compatible serialization for evaluation contexts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

import numpy as np

from chemfit.abstract_objective_function import Skipped
from chemfit.evaluate_context import (
    NO_OUTPUT,
    RESERVED_RECORD_SECTION_KEYS,
    EvaluateContext,
)

# The schema version changes only when the persistent record contract changes,
# independently of ChemFit releases.
EVALUATE_CONTEXT_SCHEMA_VERSION = 1

# Tagged values represent data that has no unambiguous strict-JSON equivalent.
_TYPE_TAG_KEY = "__type__"


class ContextSerializationError(TypeError):
    """Raised when a value cannot be serialized."""


def context_to_record(
    context: EvaluateContext,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Return a versioned, JSON-compatible record for ``context``.

    Args:
        context: Root evaluation context to serialize.
        strict: Whether unsupported values should raise
            :class:`ContextSerializationError`. Permissive mode retains them
            as tagged values instead.

    Returns:
        A schema-versioned record containing the complete evaluation hierarchy.

    """
    return {
        "format": "chemfit.evaluate_context",
        "schema_version": EVALUATE_CONTEXT_SCHEMA_VERSION,
        "chemfit_version": version("chemfit"),
        "root": serialize_context(context, strict=strict),
    }


def type_name(value: object) -> str:
    """Return a module-qualified Python type name for diagnostics."""
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def serialize_context(
    context: EvaluateContext, *, path: list[str] | None = None, strict: bool = False
) -> dict[str, Any]:
    """Serialize one context and its ordered child hierarchy."""

    # Default path is "root"
    if path is None:
        path = ["root"]

    # Status field
    if context.output is NO_OUTPUT:
        status = "no_output"
    elif isinstance(context.output, BaseException):
        status = "exception"
    elif isinstance(context.output, Skipped):
        status = "skipped"
    else:
        status = "normal"

    record_sections = context.record_sections()
    for k in record_sections:
        if k in RESERVED_RECORD_SECTION_KEYS:
            msg = f"Custom context record section contains reserved key: {k}"
            raise ContextSerializationError(msg)
        if not isinstance(k, str):
            msg = f"Custom context record section keys must be strings. Got {k}"
            raise ContextSerializationError(msg)

    return {
        "input": serialize_value(
            context.input,
            path=[*path, "input"],
            strict=strict,
        ),
        "output": serialize_value(
            context.output,
            path=[*path, "output"],
            strict=strict,
        ),
        "meta": serialize_value(
            context.meta,
            path=[*path, "meta"],
            strict=strict,
        ),
        "status": status,
        "path": ".".join(path),
        **{
            k: serialize_value(
                v,
                path=[*path, k],
                strict=strict,
            )
            for k, v in record_sections.items()
        },
        "children": serialize_children(context, path, strict=strict),
    }


def serialize_children(
    context: EvaluateContext,
    path: list[str],
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    """Serialize recorded children or known, entirely missing declared slots."""
    # Structural evaluation links all immediate children at once, in declaration
    # order. A nonempty list is therefore already complete and positionally stable.
    if context.children:
        return [
            serialize_context(child, path=[*path, str(index)], strict=strict)
            for index, child in enumerate(context.children)
        ]

    structure = context.meta.get("chemfit.structure")
    if not isinstance(structure, Mapping):
        return []

    declarations = structure.get("children")
    if not isinstance(declarations, list):
        return []

    indices = []
    for position, declaration in enumerate(declarations):
        if not isinstance(declaration, Mapping):
            return []
        index = declaration.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            return []
        indices.append(index)

    return [
        {
            "input": {_TYPE_TAG_KEY: "no_output"},
            "output": {_TYPE_TAG_KEY: "no_output"},
            "meta": {},
            "status": "not_recorded",
            "path": ".".join([*path, str(index)]),
            "children": [],
        }
        for index in indices
    ]


def serialize_value(  # noqa: PLR0911, PLR0912
    value: Any, path: list[str], *, strict: bool
) -> Any:
    """
    Convert one runtime value into its unambiguous JSON-compatible form.

    ``path`` is propagated into errors and tagged nested values.
    """

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"

    if value is None or isinstance(value, (bool, int, str, float)):
        return value

    if value is NO_OUTPUT:
        return {_TYPE_TAG_KEY: "no_output"}

    if isinstance(value, Skipped):
        return {
            _TYPE_TAG_KEY: "skipped",
            "reason": serialize_value(value.reason, [*path, "reason"], strict=strict),
        }

    if isinstance(value, BaseException):
        exception_type = type(value)
        return {
            _TYPE_TAG_KEY: "exception",
            "module": exception_type.__module__,
            "class": exception_type.__qualname__,
            "message": str(value),
        }

    if isinstance(value, np.generic):
        item = value.item()
        # fallback for numpy scalars which cannot be
        # converted to python float e.g longdouble
        if isinstance(item, np.generic):
            return {
                "__type__": "numpy_scalar",
                "dtype": str(value.dtype),
                "value": str(value),
            }
        return serialize_value(item, path, strict=strict)

    if isinstance(value, np.ndarray):
        return serialize_value(
            value.tolist(),
            path,
            strict=strict,
        )
    if isinstance(value, Mapping):
        return serialize_mapping(
            value,
            path,
            strict=strict,
        )

    if isinstance(value, (list, tuple)):
        return serialize_sequence(
            value,
            path,
            strict=strict,
        )

    if strict:
        python_type = type_name(value)
        message = f"Cannot serialize value at {'.'.join(path)}: unsupported type {python_type}"
        raise ContextSerializationError(message)

    return {
        _TYPE_TAG_KEY: "unsupported",
        "python_type": type_name(value),
        "repr": bounded_repr(value),
    }


def serialize_sequence(
    value: list[Any] | tuple[Any, ...],
    path: list[str],
    *,
    strict: bool,
) -> list[Any]:
    """Serialize a list or tuple in order, converting tuples to JSON arrays."""

    return [
        serialize_value(
            item,
            path=[*path, str(index)],
            strict=strict,
        )
        for index, item in enumerate(value)
    ]


def serialize_mapping(
    value: Mapping[str, Any],
    path: list[str],
    *,
    strict: bool,
) -> dict[str, Any]:
    """
    Serialize a mapping without colliding with reserved tagged values.

    Mappings use the natural JSON object form unless they contain the reserved
    ``__type__`` key, in which case the mapping is wrapped. Non-string keys
    are rejected because JSON object keys must be strings.
    """

    result = {}

    for key, item in value.items():
        if not isinstance(key, str):
            msg = f"Non-string dictionary key at {path}: {key!r}"
            raise ContextSerializationError(msg)

        result[key] = serialize_value(item, [*path, key], strict=strict)

    if _TYPE_TAG_KEY in result:
        return {"__type__": "mapping", "value": result}

    return result


def bounded_repr(value: object) -> str:
    """Return a defensive, size-bounded representation of an unsupported value."""
    max_length = 200
    try:
        representation = repr(value)
    except Exception as error:
        representation = f"<repr failed: {type(error).__name__}: {error}>"
    if len(representation) <= max_length:
        return representation
    return f"{representation[: max_length - 3]}..."
