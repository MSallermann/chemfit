import pytest

from chemfit.debug_utils import log_all_methods
from chemfit.utils import check_params_near_bounds


def test_check_params():
    params = {
        "electrostatic": {"bla": {"a": 1.0, "b": 1.0, "c": 1.0}, "foo": 1.0},
        "dispersion": -0.4,
        "params": {"a": 1.0, "b": 1.0},
    }

    bounds = {"dispersion": [0.2, 2.0], "electrostatic": {"bla": {"a": [0.5, 1.001]}}}

    problematic_params = check_params_near_bounds(params, bounds, relative_tol=1e-2)
    expected = [
        ("electrostatic.bla.a", 1.0, 0.5, 1.001),
        ("dispersion", -0.4, 0.2, 2.0),
    ]

    assert problematic_params == expected


def test_debug_log():
    class MyCoolObject:
        def __init__(self, a: int, b: int):
            self.a = a
            self._b = b

        def method(self, f: float, **kwargs) -> float:  # noqa: ARG002
            return f

        @property
        def b(self) -> int:
            return self._b

    log_recs = []
    obj = MyCoolObject(2, 3)
    obj_logged = log_all_methods(obj, lambda msg: log_recs.append(msg))

    obj_logged.a = 2
    obj_logged.method(3.14, bla="bla")

    assert obj_logged.a == obj.a
    assert obj_logged.b == obj.b
    assert obj_logged._b == obj.b  # noqa: SLF001

    with pytest.raises(AttributeError):
        obj_logged.b = 4  # type: ignore

    assert len(log_recs) > 0
