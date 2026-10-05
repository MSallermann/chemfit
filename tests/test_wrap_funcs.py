from __future__ import annotations

from typing import Any

import pytest

from chemfit.wrap_funcs import WrappedObjectiveFunctor, WrappedQuantityComputer


@pytest.fixture(
    params=[
        pytest.param(
            lambda: WrappedObjectiveFunctor(lambda _parameters: 0.0),
            id="objective",
        ),
        pytest.param(
            lambda: WrappedQuantityComputer(lambda _parameters: {}),
            id="quantity-computer",
        ),
    ]
)
def wrapped(request: pytest.FixtureRequest) -> Any:
    factory = request.param
    return factory()


def test_repeated_positional_bindings_accumulate(wrapped: Any) -> None:
    bound = wrapped.bind(1).bind(3)

    assert bound.func_args == (1, 3)


def test_repeated_keyword_bindings_preserve_previous_keys(wrapped: Any) -> None:
    bound = wrapped.bind(a=2).bind(b=4)

    assert bound.func_kwargs == {"a": 2, "b": 4}


def test_later_keyword_bindings_override_duplicates(wrapped: Any) -> None:
    bound = wrapped.bind(a=1).bind(a=2)

    assert bound.func_kwargs == {"a": 2}


def test_bind_does_not_change_source_wrapper(wrapped: Any) -> None:
    source = wrapped.bind(1, a=2)

    bound = source.bind(3, b=4)

    assert source.func_args == (1,)
    assert source.func_kwargs == {"a": 2}
    assert bound.func_args == (1, 3)
    assert bound.func_kwargs == {"a": 2, "b": 4}
