from functools import wraps
import inspect
from typing import Callable


def log_invocation(func, log_func: Callable[[str], None], log_args: bool = True):

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


def log_all_methods(obj, log_func: Callable[[str], None]):
    """Creates a dummy object which logs all method invocations."""

    class Dummy(object): ...

    dummy = Dummy()

    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if inspect.ismethod(attr):
            try:
                setattr(dummy, attr_name, log_invocation(attr, log_func))
            except Exception as e:
                print(f"Readonly attribute {attr_name}")

    return dummy
