import inspect
from functools import wraps
from typing import Any, Callable, TypeVar, cast


def log_invocation(
    func: Callable, log_func: Callable[[str], None], log_args: bool = True
):
    @wraps(func)
    def wrapped_with_logging(*args, **kwargs):
        log_func(f"Pre {func.__name__}")
        if log_args and len(args) > 0:
            log_func(f"    {args = }")
        if log_args and len(kwargs) > 0:
            log_func(f"    {kwargs = }")
        res = func(*args, **kwargs)
        log_func(f"Post {func.__name__}")
        return res

    return wrapped_with_logging


LoggedObject = TypeVar("LoggedObject", bound=object)


def log_all_methods(obj: LoggedObject, log_func: Callable[[str], None]) -> LoggedObject:
    """Return a proxy object that logs method calls and delegates everything to `obj`."""

    class Proxy:
        def __init__(self, wrapped: LoggedObject):
            super().__setattr__("_wrapped", wrapped)

        def __getattribute__(self, name: str):
            if name == "_wrapped":
                return super().__getattribute__(name)

            wrapped = super().__getattribute__("_wrapped")
            attr = getattr(wrapped, name)

            if inspect.ismethod(attr) or inspect.isfunction(attr):
                # Use your log_invocation directly
                return log_invocation(attr, log_func)
            return attr

        def __setattr__(self, name: str, value: Any):
            wrapped = super().__getattribute__("_wrapped")
            setattr(wrapped, name, value)

        def __delattr__(self, name: str):
            wrapped = super().__getattribute__("_wrapped")
            try:
                delattr(wrapped, name)
            except AttributeError:
                super().__delattr__(name)

        def __dir__(self):
            wrapped = super().__getattribute__("_wrapped")
            return sorted(set(dir(type(self)) + dir(wrapped)))

    return cast("LoggedObject", Proxy(obj))
