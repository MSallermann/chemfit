from pathlib import Path

import pytest

from chemfit.data_utils import process_single_csv
from chemfit.debug_utils import log_all_methods
from chemfit.executor_utils import AttachContextAsReturnValue
from chemfit.utils import check_params_near_bounds


def test_context_wrapper_requires_context_as_final_argument():
    wrapped = AttachContextAsReturnValue(lambda value: value)

    with pytest.raises(TypeError, match="final positional argument"):
        wrapped("not a context")


def test_csv_rejects_non_string_tags(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "file,tag,reference_energy\nstructure.xyz,12,0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'tag' entries must be strings"):
        process_single_csv(csv_path)


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
    obj_logged = log_all_methods(obj, log_recs.append)

    obj_logged.a = 2
    obj_logged.method(3.14, bla="bla")

    assert obj_logged.a == obj.a
    assert obj_logged.b == obj.b
    assert obj_logged._b == obj.b  # noqa: SLF001

    with pytest.raises(AttributeError):
        obj_logged.b = 4  # type: ignore

    assert len(log_recs) > 0
