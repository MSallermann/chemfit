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
        ["electrostatic.bla.a", 1.0, 0.5, 1.001],
        ["dispersion", -0.4, 0.2, 2.0],
    ]

    assert problematic_params == expected
